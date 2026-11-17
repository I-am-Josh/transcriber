# transcriber

Private, offline hold-to-talk dictation for macOS, using NVIDIA Parakeet on Apple MLX.

Hold **fn** and talk. A box at the bottom of the screen shows the words
as you say them; let go and the text is typed wherever your cursor is.

Set **System Settings → Keyboard → Press 🌐 key to** to **Do Nothing**, or macOS
will also open emoji, Apple Dictation or switch input source when you press fn.

## Open it

- **As an app:** open **Transcriber** from Applications, Launchpad or Spotlight.
  It sits in the Dock (right-click → Quit, or Cmd+Q) and adds a microphone icon
  to the menu bar with its status and *Edit word list…*. A message at the bottom
  of the screen says when it's on. Build or rebuild it with `./build_app.sh`
  (needs Xcode's command line tools for `clang`).
- **From a terminal:** `uv run dictate.py` (Ctrl+C to quit).

Run only one of them at a time, or everything is typed twice.

## Word list

`words.txt` fixes words it gets wrong, one `what it writes -> what you want` per
line (for example `Epen -> Eapen`). Changes apply to the next thing you say.

## Permissions (first run)

macOS asks for **Accessibility** (to see the key and paste) and **Microphone**.
The app waits until Accessibility is on, so there's no need to reopen it.
Rebuilding the app makes macOS ask again.

The app logs to `~/Library/Logs/Transcriber.log`.
