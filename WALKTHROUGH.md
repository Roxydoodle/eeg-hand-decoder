# How the EEG decoder works

## Start with what one trial actually is

The person saw an arrow telling them which hand to imagine moving. The EEG cap recorded voltage over time. Each trial is a recording and a label from the arrow. In Python the label is `0` for left and `1` for right. There are 72 of each so random guessing averages 50%. The original experiment also includes feet and tongue cues. The loader skips those trials and uses the 22 scalp channels. [1]

Open `decoder.py` and follow `load_trials()`. It returns three arrays. For the data here, `eeg` has shape `(144, 22, 1250)`. That means 144 trials, 22 electrodes and five seconds at 250 samples per second. `labels[i]` gives the hand for trial `i`. `runs[i]` tells you which recording block that trial came from. The raw file gives the start of each trial. The hand arrow appears two seconds after that.

## How two wiggly lines become two numbers

1. In `decoder.py` the script takes three seconds after the arrow and resamples from 250 Hz to 80 Hz. There are now 240 samples per electrode in a trial.
2. It uses a band-pass filter. With a 3 to 33 Hz band, much slower changes and much faster changes get weaker. The program keeps the filtered samples from 0.5 to 2.5 seconds after the arrow. That is the `40:200` slice in the code.
3. `log_band_power()` takes the variance of that filtered slice. Bigger swings in the band give a bigger variance. Taking the log makes a few unusually large values less dominant. For C3 and C4 you now have two numbers per trial.
4. `logistic_model()` first puts the features on a similar scale. Logistic regression learns a weight for each one. It adds the weighted values to a bias and uses the result to estimate the chance that the trial is right hand. Think of two columns in a spreadsheet with a line drawn between the two labels.

C3 is on the left side of the scalp and C4 is on the right. Imagining hand motion can reduce the power of a particular rhythm over the opposite side. A quieter rhythm means less synchronized activity at that frequency. The total voltage can still wiggle a lot. In Volunteer C the two-electrode 3 to 33 Hz model got 93 of 144 trials right. The same model using 16 to 30 Hz got 109 right. The two sensors contain some information but they throw away what nearby channels tell us. [2]

## Why the stronger version does better

One raw electrode picks up a mixture of signals. Common spatial patterns learns weighted combinations of the electrodes. Some combinations have more power for left-hand trials and others have more power for right-hand trials. The program keeps up to four combinations. LDA learns which side of a line a new set of four power values falls on. The optional version has a short model definition in `prepare_decoder()`. MNE learns the combinations and scikit-learn learns the decision rule. [3]

The recorded trials come in six blocks. In the usual run, the model learns from five blocks and predicts the sixth. Then it repeats this until every block has had a turn as the test block. The scaler or CSP gets fitted using the training trials each time. A held-out block never gives it answers while it is learning. [4]

`--tune` is closer to the way I would choose a model for a real project. It compares eleven frequency bands and three electrode sets using only recording blocks 1 to 4. The CSP run picked 5 to 30 Hz and the 13 motor-strip electrodes. It then trained on those four blocks and tested once on blocks 5 and 6. It got 21 of 24 left trials and 24 of 24 right trials correct. The score was 93.8%. The 95% Wilson interval was 83.2% to 97.9%. With only 48 test trials, the uncertainty is still wide. The printed p-value is a comparison with a simple fair-coin model.

## What changes when the program can stay still

A normal left/right classifier will always choose one hand. Run `python decoder.py --replay` for the version with two 1.5-second windows per trial. The window before the hand cue stands in for rest and the window starting at the cue is the movement attempt. The replay filters forwards only. A rest decision can only see samples that existed by the end of its window. This version uses left, right and rest as labels. It also lets uncertain predictions become rest. [5]

Raising the trigger threshold usually lowers false moves and can also silence genuine attempts. In the Python practice run the starter model gave 48 correct moves out of 96 and 22 false alarms out of 96 at threshold 0. With the threshold at 0.50 it made 38 correct moves and 16 false alarms. Every threshold missed at least one target. The replay therefore left the later trials unscored. The TKS browser simulation has its own classifier and its own rest windows. It can come to a different result even for the same participant.

The bits-per-minute figure is an estimate for two possible commands at the chosen decision speed. It assumes the decoder gets a fresh chance to send a command every window. Real communication has pauses and mistakes that need undoing. A false move at rest is its own result and can count more than a missed intended move.

## Try one change yourself

Run `python decoder.py --band 16 30`. Before you press Return, guess how many trials it will get right. Then try the same band with `--electrodes motor`. The trials stay the same so you can see what adding nearby electrodes does. In the code, start with `log_band_power()` and print `power.shape` just before the return. You should see why the model trains on one row per trial.

## Sources

1. https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf
2. https://pubmed.ncbi.nlm.nih.gov/10576479/
3. https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
5. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
