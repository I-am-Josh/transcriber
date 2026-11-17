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
import objc
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
WAVE_BARS = 20
WAVE_EVERY = 1 / 24  # seconds per bar; the bars cover the last ~0.8 s
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


def same_word(a, b):
    return re.sub(r"\W", "", a.lower()) == re.sub(r"\W", "", b.lower())


def lock_agreed(locked, previous, words):
    """Words that came out the same in two passes in a row get locked, so the
    live preview stops rewriting them; only the newest words stay unsure."""
    k = 0
    while k < min(len(words), len(previous)) and same_word(words[k], previous[k]):
        k += 1
    return locked + words[len(locked) : k] if k > len(locked) else locked


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


def loudness(rms):
    """0..1 for the wave bars: room noise stays flat, loud speech fills them."""
    db = 20 * np.log10(max(rms, 1e-6))
    return min(1.0, max(0.0, (db + 55) / 40))


class WaveView(AppKit.NSView):
    """Bars that follow how loud you're talking, newest on the right."""

    def initWithFrame_(self, frame):
        self = objc.super(WaveView, self).initWithFrame_(frame)
        if self is not None:
            self.levels = [0.0] * WAVE_BARS
        return self

    @objc.python_method
    def push(self, level):
        self.levels = self.levels[1:] + [level]
        self.setNeedsDisplay_(True)

    @objc.python_method
    def clear(self):
        self.levels = [0.0] * WAVE_BARS
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        width, height = self.bounds().size
        step = width / WAVE_BARS
        bar = step * 0.6
        AppKit.NSColor.whiteColor().set()
        for i, level in enumerate(self.levels):
            h = max(bar, level * height)
            x = i * step + (step - bar) / 2
            AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                ((x, (height - h) / 2), (bar, h)), bar / 2, bar / 2
            ).fill()


class Overlay:
    """A floating box at the bottom of the screen that never takes focus."""

    WIDTH, HEIGHT = 640, 64  # smallest size; it grows to fit the text
    PAD, WAVE_WIDTH = 18, 70
    MARGIN = 40  # gap to the screen edges once it's as wide as it gets
    MAX_HEIGHT = 0.6  # of the screen; past this the oldest words scroll off

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
        view.layer().setBackgroundColor_(Quartz.CGColorCreateGenericGray(0.08, 0.97))

        self.wave = WaveView.alloc().initWithFrame_(
            ((self.PAD, (self.HEIGHT - 32) / 2), (self.WAVE_WIDTH, 32))
        )
        self.wave.setHidden_(True)
        view.addSubview_(self.wave)

        self.label = AppKit.NSTextField.wrappingLabelWithString_("")
        self.label.setMaximumNumberOfLines_(0)
        view.addSubview_(self.label)
        self.shown = 0
        self.timer = None

    @staticmethod
    def styled(text, draft=""):
        """White text, then `draft` (words that may still change) in grey."""
        font = AppKit.NSFont.systemFontOfSize_(15)
        out = AppKit.NSMutableAttributedString.alloc().init()
        for part, alpha in ((text, 1.0), (draft, 0.45)):
            if part:
                if out.length():
                    part = " " + part
                color = AppKit.NSColor.whiteColor().colorWithAlphaComponent_(alpha)
                out.appendAttributedString_(
                    AppKit.NSAttributedString.alloc().initWithString_attributes_(
                        part, {AppKit.NSFontAttributeName: font, AppKit.NSForegroundColorAttributeName: color}
                    )
                )
        return out

    def show(self, text, draft=""):
        """Show all the text: the box widens up to the screen edges, then grows taller."""
        self.shown += 1
        screen = AppKit.NSScreen.mainScreen().visibleFrame()
        left = self.PAD + (0 if self.wave.isHidden() else self.WAVE_WIDTH + 14)
        cell = self.label.cell()
        while True:
            self.label.setAttributedStringValue_(self.styled(text, draft))
            one_line = cell.cellSizeForBounds_(((0, 0), (1e5, 1e5))).width
            width = min(max(self.WIDTH, left + one_line + self.PAD + 4), screen.size.width - 2 * self.MARGIN)
            text_width = width - left - self.PAD
            text_height = cell.cellSizeForBounds_(((0, 0), (text_width, 1e5))).height
            if text_height + 2 * self.PAD <= screen.size.height * self.MAX_HEIGHT or len(text.split()) < 2:
                break
            text = "…" + " ".join(text.split()[max(1, len(text.split()) // 10):])
        height = max(self.HEIGHT, text_height + 2 * self.PAD)
        x = screen.origin.x + (screen.size.width - width) / 2
        self.panel.setFrame_display_(((x, screen.origin.y + 60), (width, height)), True)
        self.label.setFrame_(((left, (height - text_height) / 2), (text_width, text_height)))
        self.wave.setFrameOrigin_((self.PAD, height - (self.HEIGHT + 32) / 2))  # level with the first line
        if not self.panel.isVisible():
            self.panel.orderFrontRegardless()

    def listen(self, take_level):
        """Show moving waves while you hold fn; take_level() is the loudness since last call."""
        self.wave.clear()
        self.wave.setHidden_(False)
        self.show("Listening…")
        if self.timer is None:
            self.timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                WAVE_EVERY, True, lambda timer: self.wave.push(loudness(take_level()))
            )

    def stop_waves(self):
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None
        self.wave.clear()

    def notice(self, text, seconds=3):
        """Show a message, then hide it unless something newer replaced it."""
        self.stop_waves()
        self.wave.setHidden_(True)
        self.show(text)
        token = self.shown
        AppHelper.callLater(seconds, lambda: self.shown == token and self.hide())

    def hide(self):
        self.stop_waves()
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
        self.level = 0.0  # loudest chunk since the waves last looked

    # Microphone thread
    def on_audio(self, indata, frames, t, status):
        if self.recording:
            chunk = indata[:, 0].copy()
            self.audio.put(chunk)
            self.level = max(self.level, float(np.sqrt(np.mean(chunk**2))))

    # Main thread
    def take_level(self):
        level, self.level = self.level, 0.0
        return level

    # Main thread
    def on_flags_changed(self, event):
        if event.keyCode() != FN_KEYCODE:
            return
        down = bool(event.modifierFlags() & FN_FLAG)
        if down and not self.recording:
            self.recording = True
            self.menu.set_listening(True)
            self.overlay.listen(self.take_level)  # right away, even while the model loads
            self.events.put("start")
        elif not down and self.recording:
            self.recording = False
            self.menu.set_listening(False)
            self.overlay.stop_waves()
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
                model = None  # free the model until the next time you talk
                gc.collect()
                continue
            if event != "start":
                continue
            if model is None:
                model = from_pretrained(MODEL)  # the mic keeps recording meanwhile
            fixes = load_words()
            heard = np.zeros(0, dtype=np.float32)
            onset = None  # where speech starts in `heard`
            locked, previous = [], []  # live preview: settled words, last pass
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
                if stopping or len(heard) - onset < MIN_SECONDS * SAMPLE_RATE:
                    continue
                # Re-read everything said so far, the same way as the final pass,
                # so the preview matches what gets typed.
                words = transcribe(model, heard[onset:]).split()
                # Each pass has a new length, so MLX can't reuse its scratch
                # buffers; without this they pile up (3 GB after 10 s of talking).
                mx.clear_cache()
                locked = lock_agreed(locked, previous, words)
                previous = words
                draft = words[len(locked) :]
                if words:
                    self.ui(
                        self.overlay.show,
                        fix_words(" ".join(locked), fixes),
                        fix_words(" ".join(draft), fixes),
                    )

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
