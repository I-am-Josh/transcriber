"""Hold-to-talk dictation that runs entirely on this Mac.

Hold Right Option and speak: a small box at the bottom of the screen shows the
words as you say them. Let go and the text is typed wherever your cursor is.
Run with:  uv run dictate.py
"""

import os
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave

# Once the model is downloaded, load it from disk without asking Hugging Face.
_CACHE = pathlib.Path.home() / ".cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3"
if _CACHE.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import AppKit
import HIServices
import mlx.core as mx
import Quartz
import numpy as np
import sounddevice as sd
from parakeet_mlx import from_pretrained
from PyObjCTools import AppHelper

WORDS_FILE = pathlib.Path(__file__).with_name("words.txt")
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
RIGHT_OPTION_KEYCODE = 61
RIGHT_OPTION_FLAG = 0x40  # device-specific bit, so Left Option doesn't count
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.3  # ignore accidental taps
LIVE_EVERY = 0.25  # seconds between live updates
PREVIEW_CHARS = 110
# Speech starts when two 50 ms frames in a row are louder than this. Silence and
# the key's click make the model guess words like "yeah", so they're skipped.
# Raise it if room noise still sneaks in; lower it if quiet speech is missed.
SPEECH_LEVEL = 0.01
FRAME = SAMPLE_RATE // 20
PRE_ROLL = int(0.15 * SAMPLE_RATE)  # keep the soft start of the first word


def speech_start(audio):
    """Sample index where speech begins, or None if it's all quiet so far."""
    n = len(audio) // FRAME
    if n < 2:
        return None
    rms = np.sqrt(np.mean(audio[: n * FRAME].reshape(n, FRAME) ** 2, axis=1))
    loud = rms > SPEECH_LEVEL
    hits = np.flatnonzero(loud[:-1] & loud[1:])
    return max(0, hits[0] * FRAME - PRE_ROLL) if len(hits) else None


def load_words():
    """Read `wrong -> right` lines from words.txt."""
    fixes = []
    try:
        lines = WORDS_FILE.read_text().splitlines()
    except FileNotFoundError:
        return fixes
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "->" not in line:
            continue
        wrong, right = (part.strip() for part in line.split("->", 1))
        if wrong:
            pattern = re.compile(r"(?<!\w)" + re.escape(wrong).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
            fixes.append((pattern, right))
    return fixes


def fix_words(text, fixes):
    for pattern, right in fixes:
        text = pattern.sub(right, text)
    return text


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def transcribe(model, audio):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        write_wav(path, audio)
        return model.transcribe(path).text.strip()
    finally:
        os.remove(path)


def type_text(text):
    """Paste text at the cursor, then put the old clipboard back."""
    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode())
    subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
        check=False,
    )
    time.sleep(0.3)
    subprocess.run(["pbcopy"], input=old)


class Overlay:
    """A floating box at the bottom of the screen that never takes focus."""

    WIDTH, HEIGHT = 560, 64

    def __init__(self):
        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        self.panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (self.WIDTH, self.HEIGHT)), style, AppKit.NSBackingStoreBuffered, False
        )
        self.panel.setLevel_(AppKit.NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        view = self.panel.contentView()
        view.setWantsLayer_(True)
        view.layer().setCornerRadius_(10)
        view.layer().setBackgroundColor_(Quartz.CGColorCreateGenericGray(0.08, 0.9))

        self.label = AppKit.NSTextField.wrappingLabelWithString_("")
        self.label.setFrame_(((18, 10), (self.WIDTH - 36, self.HEIGHT - 20)))
        self.label.setFont_(AppKit.NSFont.systemFontOfSize_(15))
        self.label.setTextColor_(AppKit.NSColor.whiteColor())
        self.label.setMaximumNumberOfLines_(2)
        view.addSubview_(self.label)

    def show(self, text):
        if not self.panel.isVisible():
            screen = AppKit.NSScreen.mainScreen().visibleFrame()
            x = screen.origin.x + (screen.size.width - self.WIDTH) / 2
            self.panel.setFrameOrigin_((x, screen.origin.y + 60))
            self.panel.orderFrontRegardless()
        if len(text) > PREVIEW_CHARS:
            text = "…" + text[-PREVIEW_CHARS:].lstrip()
        self.label.setStringValue_("● " + text)

    def hide(self):
        self.panel.orderOut_(None)


class Dictation:
    """All MLX work happens on one worker thread; the main thread runs the UI."""

    def __init__(self, overlay):
        self.overlay = overlay
        self.recording = False
        self.events = queue.Queue()  # "start" / "stop"
        self.audio = queue.Queue()  # chunks from the microphone

    # Microphone thread
    def on_audio(self, indata, frames, t, status):
        if self.recording:
            self.audio.put(indata[:, 0].copy())

    # Main thread
    def on_flags_changed(self, event):
        if event.keyCode() != RIGHT_OPTION_KEYCODE:
            return
        down = bool(event.modifierFlags() & RIGHT_OPTION_FLAG)
        if down and not self.recording:
            self.recording = True
            self.events.put("start")
        elif not down and self.recording:
            self.recording = False
            self.events.put("stop")

    def ui(self, method, *args):
        AppHelper.callAfter(method, *args)

    def drain_audio(self):
        chunks = []
        while True:
            try:
                chunks.append(self.audio.get_nowait())
            except queue.Empty:
                return chunks

    # Worker thread
    def run(self):
        print("Loading model…")
        model = from_pretrained(MODEL)
        transcribe(model, np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm up
        print("Ready. Hold Right Option and talk. Ctrl+C to quit.")

        while True:
            if self.events.get() != "start":
                continue
            self.ui(self.overlay.show, "Listening…")
            fixes = load_words()
            heard = np.zeros(0, dtype=np.float32)
            onset = None  # where speech starts in `heard`
            fed = 0  # how much of `heard` the live preview has seen
            with model.transcribe_stream(context_size=(256, 256)) as stream:
                stopping = False
                while not stopping:
                    try:
                        stopping = self.events.get(timeout=LIVE_EVERY) == "stop"
                    except queue.Empty:
                        pass
                    chunks = self.drain_audio()
                    if not chunks:
                        continue
                    heard = np.concatenate([heard, *chunks])
                    if onset is None:
                        onset = speech_start(heard)
                        if onset is None:
                            continue
                        fed = onset
                    if not stopping:
                        stream.add_audio(mx.array(heard[fed:]))
                        fed = len(heard)
                        live = fix_words(stream.result.text.strip(), fixes)
                        if live:
                            self.ui(self.overlay.show, live)

            audio = heard[onset:] if onset is not None else heard[:0]
            if len(audio) < MIN_SECONDS * SAMPLE_RATE:
                self.ui(self.overlay.hide)
                continue
            # One full pass over the whole recording is a little more accurate
            # than the live preview.
            start = time.time()
            text = fix_words(transcribe(model, audio), fixes)
            self.ui(self.overlay.hide)
            if text:
                type_text(text + " ")
                print(f"[{time.time() - start:.2f}s] {text}")


def check_permissions():
    """Without Accessibility, macOS hides key presses and blocks pasting."""
    prompt = {HIServices.kAXTrustedCheckOptionPrompt: True}
    if HIServices.AXIsProcessTrustedWithOptions(prompt):
        return
    app = os.environ.get("TERM_PROGRAM", "your terminal app").replace("Apple_Terminal", "Terminal")
    print(
        f"{app} needs Accessibility permission.\n"
        f"  System Settings → Privacy & Security → Accessibility → turn on {app}\n"
        f"  (also Input Monitoring, if {app} is listed there).\n"
        f"Then quit {app} completely (Cmd+Q), reopen it, and run this again."
    )
    sys.exit(1)


def main():
    check_permissions()
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon

    dictation = Dictation(Overlay())
    threading.Thread(target=dictation.run, daemon=True).start()

    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=dictation.on_audio)
    mic.start()
    monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        AppKit.NSEventMaskFlagsChanged, dictation.on_flags_changed
    )
    try:
        AppHelper.runEventLoop(installInterrupt=True)
    finally:
        AppKit.NSEvent.removeMonitor_(monitor)
        mic.stop()


if __name__ == "__main__":
    main()
