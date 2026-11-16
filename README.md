# transcriber

Private, offline hold-to-talk dictation for macOS, using NVIDIA Parakeet on Apple MLX.

## Run

    uv run dictate.py

Hold **Right Option**, talk, let go. The text is typed wherever your cursor is.

## First-run permissions (System Settings → Privacy & Security)

Give the terminal app you run this from:
- **Microphone**: to record
- **Input Monitoring**: to see the Right Option key
- **Accessibility**: to paste the text

Restart the terminal after granting them.
