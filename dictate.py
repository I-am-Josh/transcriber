"""Hold-to-talk dictation that runs entirely on this Mac.

Hold Right Option, speak, let go: the words are typed wherever your cursor is.
Run with:  uv run dictate.py
"""

import os
import queue
import subprocess
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd
from parakeet_mlx import from_pretrained
from pynput import keyboard

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
HOTKEY = keyboard.Key.alt_r  # Right Option
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.3  # ignore accidental taps


def play(sound):
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"])


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


def main():
    print("Loading model…")
    model = from_pretrained(MODEL)
    transcribe(model, np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm up

    chunks = []
    recording = False
    jobs = queue.Queue()

    def on_audio(indata, frames, t, status):
        if recording:
            chunks.append(indata[:, 0].copy())

    def on_press(key):
        nonlocal recording
        if key == HOTKEY and not recording:
            chunks.clear()
            recording = True
            play("Tink")

    def on_release(key):
        nonlocal recording
        if key == HOTKEY and recording:
            recording = False
            play("Pop")
            if chunks:
                jobs.put(np.concatenate(chunks))

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    with stream, listener:
        print("Ready. Hold Right Option and talk. Ctrl+C to quit.")
        # MLX work stays on the main thread; the key listener only queues audio.
        while True:
            audio = jobs.get()
            if len(audio) < MIN_SECONDS * SAMPLE_RATE:
                continue
            start = time.time()
            text = transcribe(model, audio)
            if text:
                type_text(text + " ")
                print(f"[{time.time() - start:.2f}s] {text}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
