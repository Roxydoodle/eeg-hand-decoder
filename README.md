# EEG hand decoder

Can a program tell which hand someone imagined moving from signals on their scalp? This is a single-file Python version of the TKS EEG Decoder Lab with Volunteer C. It uses the lab's examples and was built with ChatGPT assistance.

The first model uses logistic regression and two electrodes. Start there if you want to see what the model is actually reading. The stronger version uses common spatial patterns (CSP) with linear discriminant analysis (LDA), the same two methods as the browser lab.

| What I ran in Python | Result |
| --- | --- |
| C3 and C4 with logistic regression at 3 to 33 Hz | 64.6% (93/144) with whole recording runs held out |
| C3 and C4 with logistic regression at 16 to 30 Hz | 75.7% (109/144) in the same exploratory comparison |
| CSP + LDA chosen on early recordings | 93.8% (45/48) on the two later runs |

The plots and results in the browser lab come from its own five-fold split. The Python project uses whole recording runs as groups. Both evaluations come from one day for one participant.

## Run it

Use Python 3.9 or newer. These commands are for macOS or Linux and should be run from the project folder.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python decoder.py
```

The first run downloads roughly 45 MB from Graz University of Technology. Later runs use the same checked file in `data/`.

```sh
python decoder.py --compare
python -m pip install 'mne>=1.7,<2'
python decoder.py --model csp --tune
python decoder.py --replay
```

- `--compare` tries mu only, beta only and the back of the head. Each changes one setting from the baseline.
- `--model csp --tune` compares 33 combinations on the first four runs. It picks one combination and evaluates it on the last two.
- `--replay` gives the decoder a third option, stay still. It uses the time before the arrow as a stand-in for rest and checks whether any trigger setting meets the lab targets in practice before scoring later trials.

All the Python is in `decoder.py` [1]. To follow one trial through the program, start with `load_trials()`, then read `log_band_power()` and `held_out_predictions()`. Walkthrough [2].

## Data and credit

The recordings are from BCI Competition IV data set 2a, published by C. Brunner, R. Leeb, G. R. Müller-Putz, A. Schlögl and G. Pfurtscheller at Graz University of Technology. The file is `A09T.mat`. Six visible traces from Volunteer C matched this file when compared against the nine published training sessions. The 144 left/right trials come from six blocks in one session. [3] [4] [5]

Graz lists the dataset under CC BY-ND 4.0. The repository contains the source code and downloads the original file into the ignored `data/` folder. [6]

The channel order follows the MOABB dataset loader [7]. The CSP example follows the model sequence from the TKS EEG Decoder Lab [8] and MNE's motor imagery tutorial [9].

## Getting help with setup

If you’re new to Python, copy the prompt in `prompt.md` [10] into a coding assistant that can use a workspace and terminal. It can download the project, set up Python in the project folder and explain the result of the first run.

## Sources

1. [decoder.py](decoder.py)
2. [WALKTHROUGH.md](WALKTHROUGH.md)
3. https://bnci-horizon-2020.eu/database/data-sets
4. https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf
5. https://doi.org/10.3389/fnins.2012.00055
6. https://creativecommons.org/licenses/by-nd/4.0/
7. https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py
8. https://claude.ai/artifact/D8U2DFjC4oH1fxUSweGHqu
9. https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
10. [prompt.md](prompt.md)
