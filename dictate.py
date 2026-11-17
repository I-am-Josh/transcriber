"""Hold-to-talk dictation that runs entirely on this Mac.

Hold fn and speak: a small box at the bottom of the screen shows the
words as you say them. Let go and the text is typed wherever your cursor is.
Run with:  uv run dictate.py   (or open Transcriber.app, see build_app.sh)
"""

import gc
import os
import pathlib
import queue
import re
import signal
import sys
import subprocess
import threading
import traceback
import time

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
from parakeet_mlx.audio import get_logmel
from PyObjCTools import AppHelper

WORDS_FILE = pathlib.Path(__file__).with_name("words.txt")
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
FN_KEYCODE = 63  # the fn / Globe key
FN_FLAG = AppKit.NSEventModifierFlagFunction
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.3  # ignore accidental taps
LIVE_EVERY = 0.25  # seconds between live updates
PREVIEW_CHARS = 110
# Seconds to keep the speech model (~1.2 GB) in memory after you finish talking.
# 0 frees it straight away; it reloads in under a second next time you hold fn,
# while the mic already records, so nothing you say is lost.
KEEP_MODEL_LOADED = 0
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


def transcribe(model, audio):
    """Same as model.transcribe(file), but straight from memory, so no ffmpeg."""
    mel = get_logmel(mx.array(audio, dtype=mx.float32), model.preprocessor_config)
    return model.generate(mel)[0].text.strip()


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
        self.shown = 0

    def show(self, text, marker="● "):
        self.shown += 1
        if not self.panel.isVisible():
            screen = AppKit.NSScreen.mainScreen().visibleFrame()
            x = screen.origin.x + (screen.size.width - self.WIDTH) / 2
            self.panel.setFrameOrigin_((x, screen.origin.y + 60))
            self.panel.orderFrontRegardless()
        if len(text) > PREVIEW_CHARS:
            text = "…" + text[-PREVIEW_CHARS:].lstrip()
        self.label.setStringValue_(marker + text)

    def notice(self, text, seconds=3):
        """Show a message, then hide it unless something newer replaced it."""
        self.show(text, marker="")
        token = self.shown
        AppHelper.callLater(seconds, lambda: self.shown == token and self.hide())

    def hide(self):
        self.panel.orderOut_(None)


ACCESSIBILITY_SETTINGS = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"


class MenuActions(AppKit.NSObject):
    def editWords_(self, sender):
        subprocess.run(["open", "-t", str(WORDS_FILE)])

    def openAccessibility_(self, sender):
        subprocess.run(["open", ACCESSIBILITY_SETTINGS])


class AppDelegate(AppKit.NSObject):
    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_windows):
        # Opening the app again (Dock, Spotlight) while it runs: say it's on.
        self.dictation.announce()
        return False


def build_main_menu(actions):
    """The Transcriber menu at the top left, with Quit (Cmd+Q)."""
    main = AppKit.NSMenu.alloc().init()
    app_item = main.addItemWithTitle_action_keyEquivalent_("Transcriber", None, "")
    menu = AppKit.NSMenu.alloc().initWithTitle_("Transcriber")
    words = menu.addItemWithTitle_action_keyEquivalent_("Edit word list…", "editWords:", "")
    words.setTarget_(actions)
    menu.addItem_(AppKit.NSMenuItem.separatorItem())
    menu.addItemWithTitle_action_keyEquivalent_("Quit Transcriber", "terminate:", "q")
    app_item.setSubmenu_(menu)
    return main


class MenuBar:
    """Microphone icon in the menu bar: status, word list, quit."""

    def __init__(self):
        self.actions = MenuActions.alloc().init()
        self.item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
            AppKit.NSVariableStatusItemLength
        )
        self.set_listening(False)

        menu = AppKit.NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)
        self.status = menu.addItemWithTitle_action_keyEquivalent_("Loading…", None, "")
        self.status.setEnabled_(False)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        words = menu.addItemWithTitle_action_keyEquivalent_("Edit word list…", "editWords:", "")
        words.setTarget_(self.actions)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        menu.addItemWithTitle_action_keyEquivalent_("Quit Transcriber", "terminate:", "q")
        self.item.setMenu_(menu)

    def set_listening(self, listening):
        name = "mic.fill" if listening else "mic"
        image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Transcriber")
        image.setTemplate_(True)
        self.item.button().setImage_(image)

    def set_status(self, text, opens_settings=False):
        self.status.setTitle_(text)
        self.status.setEnabled_(opens_settings)
        self.status.setAction_("openAccessibility:" if opens_settings else None)
        self.status.setTarget_(self.actions if opens_settings else None)


class Dictation:
    """All MLX work happens on one worker thread; the main thread runs the UI."""

    def __init__(self, overlay, menu):
        self.overlay = overlay
        self.menu = menu
        self.recording = False
        self.status = "Starting Transcriber…"
        self.events = queue.Queue()  # "start" / "stop"
        self.audio = queue.Queue()  # chunks from the microphone

    # Microphone thread
    def on_audio(self, indata, frames, t, status):
        if self.recording:
            self.audio.put(indata[:, 0].copy())

    # Main thread
    def on_flags_changed(self, event):
        if event.keyCode() != FN_KEYCODE:
            return
        down = bool(event.modifierFlags() & FN_FLAG)
        if down and not self.recording:
            self.recording = True
            self.menu.set_listening(True)
            self.events.put("start")
        elif not down and self.recording:
            self.recording = False
            self.menu.set_listening(False)
            self.events.put("stop")

    def ui(self, method, *args):
        AppHelper.callAfter(method, *args)

    def set_status(self, text, opens_settings=False, announce=True):
        self.status = text
        self.ui(self.menu.set_status, text, opens_settings)
        if announce:
            self.ui(self.announce)

    # Main thread
    def announce(self):
        if not self.recording:
            self.overlay.notice(self.status, seconds=4)

    def drain_audio(self):
        chunks = []
        while True:
            try:
                chunks.append(self.audio.get_nowait())
            except queue.Empty:
                return chunks

    # Worker thread
    def run(self):
        try:
            self.work()
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            self.set_status("Transcriber stopped because of an error. Quit and open it again.")

    def work(self):
        self.wait_for_accessibility()
        self.set_status("Starting Transcriber…")
        print("Loading model…", flush=True)
        model = from_pretrained(MODEL)  # downloads it the first time
        transcribe(model, np.zeros(SAMPLE_RATE, dtype=np.float32))  # check it works
        self.set_status("Transcriber is on. Hold fn and talk.")
        print("Ready. Hold fn and talk.", flush=True)

        while True:
            # Hand back MLX's spare scratch memory (~0.8 GB) while idle.
            mx.clear_cache()
            try:
                timeout = KEEP_MODEL_LOADED if model is not None else None
                event = self.events.get(timeout=timeout)
            except queue.Empty:
                model = stream = None  # free the model until the next time you talk
                gc.collect()
                continue
            if event != "start":
                continue
            self.ui(self.overlay.show, "Listening…")
            if model is None:
                model = from_pretrained(MODEL)  # the mic keeps recording meanwhile
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
                print(f"[{time.time() - start:.2f}s] {text}", flush=True)


    def wait_for_accessibility(self):
        """Without Accessibility, macOS hides key presses and blocks pasting.

        Asks once (macOS shows its own prompt), then waits until it's allowed.
        """
        prompt = {HIServices.kAXTrustedCheckOptionPrompt: True}
        if not HIServices.AXIsProcessTrustedWithOptions(prompt):
            self.set_status("Turn on Transcriber in System Settings → Accessibility.", True)
            print(
                "Needs Accessibility permission: System Settings → Privacy & Security →\n"
                "Accessibility → turn on this app (Transcriber, or your terminal).",
                flush=True,
            )
            while not HIServices.AXIsProcessTrusted():
                time.sleep(1)
        self.ui(self.listen_for_hotkey)

    # Main thread
    def listen_for_hotkey(self):
        self.monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            AppKit.NSEventMaskFlagsChanged, self.on_flags_changed
        )


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C quits at once
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)  # Dock icon

    menu_bar = MenuBar()
    dictation = Dictation(Overlay(), menu_bar)
    delegate = AppDelegate.alloc().init()
    delegate.dictation = dictation
    app.setDelegate_(delegate)
    app.setMainMenu_(build_main_menu(menu_bar.actions))
    threading.Thread(target=dictation.run, daemon=True).start()

    mic = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=dictation.on_audio)
    mic.start()
    try:
        AppHelper.runEventLoop()
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
