# Run the EEG decoder with an AI coding assistant

Copy the text below into Codex, Claude Code or another coding assistant with access to a workspace and terminal. If you’re using a regular chat, it can guide you through the same steps.

```text
Help me set up and run the EEG hand decoder at https://github.com/Roxydoodle/eeg-hand-decoder.

I'm new to Python. Explain each step in plain language and tell me what actually worked.

1. Use my current workspace. If the repo is already here, use that copy. Otherwise clone it from the link above. Read README.md, requirements.txt and decoder.py before running it.
2. Check that I have Python 3.9 or newer. Create a .venv in the project folder and use its Python to install requirements.txt. Use the commands for my operating system.
3. Run decoder.py with no extra options. The first run downloads about 45 MB from Graz University of Technology. It saves the recording in data/ and checks the file before using it.
4. Show me the result from my run. Explain in a few sentences what the score means and where in decoder.py the program makes its guess. Point me to the README if I want to try the other experiments.

If the university download times out, explain that the recording is the part that could not download. If I already have the original A09T.mat, put it in data/ or pass its folder with --data-dir and run again. If I need to download it myself, show me the original download link from decoder.py.

If you can't use a terminal, give me one command at a time and tell me where to run it. If something else fails, show me the error in plain language and help me get past it.
```
