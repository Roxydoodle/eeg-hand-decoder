# eeg hand decoder for volunteer c
# based on the tks lab examples and built with chatgpt help

import argparse  # read the settings given when the script starts
from hashlib import sha256  # check the fingerprint of the downloaded file
from math import log2  # work out the speed estimate in bits
from pathlib import Path  # join folder and file names
from shutil import copyfileobj  # stream the download into a file
from tempfile import NamedTemporaryFile  # hold the download until it is ready
from urllib.request import urlopen  # open the download link

import numpy as np  # hold and work with the eeg arrays
from scipy.io import loadmat  # open the matlab recording
# scipy does the filtering and changing the sample rate
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from scipy.stats import binomtest  # compare the score with chance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # pick a hand from csp features
from sklearn.linear_model import LogisticRegression  # the basic model
# use whole recording blocks for the training and test split
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline  # join steps that should train together
from sklearn.preprocessing import StandardScaler  # put features on a similar scale

# start with the original graz recording for volunteer c
DATA_DIR = Path(__file__).resolve().parent / "data"  # save beside the script
FILE_NAME = "A09T.mat"
DATA_URL = "https://lampx.tugraz.at/~bci/database/001-2014/A09T.mat"
# this is the fingerprint of the graz file we ran the experiments on
FILE_HASH = "b28d8a262c779c8cad9cc80ee6aa9c5691cfa6617c03befe490a090347ebd15c"
SAMPLE_RATE = 250  # samples per sec in the original recording
CUE_SAMPLE = 2 * SAMPLE_RATE  # arrow comes up two seconds into the trial
ANALYSIS_RATE = 80  # samples per sec in the offline decoder
DEFAULT_BAND = (3, 33)  # low and high ends of the starting filter in hz

# the first 22 columns are scalp eeg in this order
CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
# named groups so we can change which sensors the model sees
CHANNEL_SETS = {
    "hands": ("C3", "C4"),  # just the two side electrodes for the basic test
    # more electrodes around the movement areas
    "motor": (
        "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CPz", "CP4",
    ),
    "back": ("P1", "Pz", "P2", "POz"),  # use sensors nearer the back as a comparison
    "all": CHANNELS,  # all 22 scalp electrodes
}
# try these low and high cutoffs when looking for a better setup
SEARCH_BANDS = (
    DEFAULT_BAND, (4, 30), (5, 30), (6, 30), (7, 25), (7, 30),
    (8, 13), (8, 25), (8, 30), (10, 30), (16, 30),
)


def get_file(data_dir):
    # this lets the caller give a folder name or a path
    data_dir = Path(data_dir)
    path = data_dir / FILE_NAME  # where the finished recording should be
    if path.is_file():
        # check the saved file so we use the same data each time
        if sha256(path.read_bytes()).hexdigest() != FILE_HASH:
            raise ValueError(f"{path} does not match the published file used here.")
        return path  # use the copy we already have

    # make the folder if this is the first run
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FILE_NAME} from Graz University of Technology (~44 MB).")
    # download into a temp file so a stopped download never looks complete
    with NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as output:
        # remember where to find it after the temp handle closes
        temp_path = Path(output.name)
    try:
        # open the download and copy it over in chunks
        with urlopen(DATA_URL, timeout=60) as response, temp_path.open("wb") as output:
            copyfileobj(response, output)
        # check the downloaded bytes before giving it the real name
        if sha256(temp_path.read_bytes()).hexdigest() != FILE_HASH:
            raise ValueError("The downloaded file did not match the expected SHA-256.")
        temp_path.replace(path)
    finally:
        # clean up the temp copy if anything went wrong
        temp_path.unlink(missing_ok=True)
    return path


def load_trials(data_dir=DATA_DIR):
    # the mat file stores the samples and cues one recording block at a time
    blocks = loadmat(get_file(data_dir), simplify_cells=True)["data"]
    # save a trace its answer and its block number together
    eeg, labels, runs = [], [], []
    run_number = 0
    for block in blocks:
        if len(block["trial"]) == 0:
            continue  # eye calibration block with no hand trials
        run_number += 1
        # our times only line up with the samples if the rate is right
        if block["fs"] != SAMPLE_RATE:
            raise ValueError("Expected 250 samples per second.")

        # pair each start position with the cue the person saw
        for start, label in zip(block["trial"], block["y"]):
            if label not in (1, 2):
                continue  # skip the feet and tongue cues
            start = int(start) - 1  # matlab counts from one and python counts from zero
            end = start + 5 * SAMPLE_RATE  # take five seconds from trial start
            # two seconds before the arrow and three after it
            # drop eye channels and put electrodes first with the transpose
            trial = block["X"][start:end, :len(CHANNELS)].T
            # every trial needs the same number of electrodes and samples
            if trial.shape != (len(CHANNELS), 5 * SAMPLE_RATE):
                raise ValueError("A trial is missing EEG samples.")
            # skip training entirely if there is a missing or infinite value
            if not np.isfinite(trial).all():
                raise ValueError("A trial has an invalid EEG sample.")
            eeg.append(trial)
            labels.append(int(label) - 1)  # now zero is left and one is right
            runs.append(run_number)  # keep the block number for the test split

    # check we got all six blocks and the expected count for each hand
    if run_number != 6 or np.bincount(labels, minlength=2).tolist() != [72, 72]:
        raise ValueError("Expected six runs with 72 left and 72 right trials in total.")
    # one row per trial and the labels and run numbers line up with those rows
    return np.stack(eeg), np.array(labels), np.array(runs)


def pick_channels(eeg, electrodes):
    # turn each sensor name into its position in the recording
    indices = [CHANNELS.index(name) for name in CHANNEL_SETS[electrodes]]
    return eeg[:, indices, :]  # keep all trials and times but only these electrodes


def log_band_power(filtered_eeg):
    # bigger swings after filtering mean more power in the band
    # the last axis is time and var turns that whole trace into one number
    power = np.var(filtered_eeg, axis=-1)
    # one number per electrode now and the tiny minimum keeps log away from zero
    return np.log(np.maximum(power, 1e-12))


def logistic_model():
    # scale the numbers and learn which hand they point to
    # the solver gets up to a thousand steps to find the weights
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def prepare_decoder(trials, band, electrodes, model_name):
    # only give the filter the sensors for this experiment
    eeg = pick_channels(trials, electrodes)
    # keep the chosen range of rhythms then take the middle two seconds
    # sos stores the filter as smaller steps that work together
    sections = butter(4, band, btype="bandpass", fs=ANALYSIS_RATE, output="sos")
    filtered = sosfiltfilt(sections, eeg, axis=-1)  # filter both ways for offline use
    window = filtered[:, :, 40:200]  # half a sec to two and a half sec after the arrow

    if model_name == "logistic":
        # the default gives the model just two numbers for each trial
        return logistic_model(), log_band_power(window)

    # only the csp version needs mne so the basic run can skip it
    try:
        from mne import set_log_level
        from mne.decoding import CSP
    except ImportError as error:  # tell the reader what to install if mne is missing
        raise SystemExit("Install MNE first with: python -m pip install mne") from error
    set_log_level("ERROR")  # keep the extra training messages out of the scores
    # csp learns up to four mixes of electrodes that differ for left and right
    components = min(4, window.shape[1])
    # lda uses the power of those mixes to choose a hand
    model = make_pipeline(
        CSP(n_components=components, log=True),  # first get one power feature per mix
        LinearDiscriminantAnalysis(),  # then learn where to split left from right
    )
    return model, window  # csp needs the traces so it can learn the mixes in training


def held_out_predictions(model, features, labels, runs):
    # fit on the other whole blocks and predict the one left out
    # sklearn fits the scaler or csp again for each split
    # the result is one guess per trial in the original order
    return cross_val_predict(model, features, labels, groups=runs, cv=LeaveOneGroupOut())


def show_result(name, band, electrodes, labels, predictions):
    matches = predictions == labels  # true for each correct guess
    correct = int(matches.sum())  # true counts as one and false counts as zero
    accuracy = matches.mean()  # divide all the right guesses by the number of trials
    # compare with coin flips and show how uncertain the accuracy is
    # the p value asks how often fair guesses would get this many right or more
    chance_test = binomtest(correct, len(labels), p=0.5, alternative="greater")
    # the interval gives a sense of the uncertainty around the score
    interval = binomtest(correct, len(labels)).proportion_ci(method="wilson")
    # print the setup followed by its overall score
    print(f"\n{name} | {band[0]:g} to {band[1]:g} Hz | {electrodes}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(labels)})")
    print(f"95% Wilson interval: {interval.low:.1%} to {interval.high:.1%}")
    print(f"One-sided binomial p vs 50%: {chance_test.pvalue:.3g}")
    # split the results by hand to see if one side is harder for this model
    for label, hand in [(0, "Left"), (1, "Right")]:
        hand_matches = matches[labels == label]  # count this hand on its own
        print(f"{hand}: {hand_matches.sum()}/{len(hand_matches)} correct")
    return accuracy  # the comparison run needs this to show how much the score moved


def tune_on_early_runs(trials, labels, runs, model_name):
    # only the first four blocks help pick the band and electrodes
    early = runs <= 4
    best_accuracy = -1  # start below any real score so the first setup gets saved
    # test every band with each of these three groups of electrodes
    for band in SEARCH_BANDS:
        for electrodes in ("hands", "motor", "all"):
            # only use the early recordings for the search
            model, features = prepare_decoder(trials[early], band, electrodes, model_name)
            guesses = held_out_predictions(model, features, labels[early], runs[early])
            accuracy = np.mean(guesses == labels[early])
            # keep the first setup if two scores tie
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_band, best_electrodes = band, electrodes

    # train with the winning settings then try the two later blocks once
    print(f"Compared {3 * len(SEARCH_BANDS)} settings on runs 1 to 4.")
    print(f"Locked {best_band[0]} to {best_band[1]} Hz | {best_electrodes}")
    print(f"Best early cross-validation accuracy: {best_accuracy:.1%}")
    # make features with the chosen setup and fit with the early labels only
    model, features = prepare_decoder(trials, best_band, best_electrodes, model_name)
    model.fit(features[early], labels[early])
    guesses = model.predict(features[~early])  # tilde means the rows outside early
    show_result("Later test (runs 5 and 6)", best_band, best_electrodes, labels[~early], guesses)


def run_offline(eeg, labels, runs, args, band, electrodes):
    # only the part after the arrow and change the sample rate from 250 to 80
    # the ratio of the new rate to the old one is 8 over 25
    trials = resample_poly(eeg[:, :, CUE_SAMPLE:], up=8, down=25, axis=-1)
    print(f"A09T | {len(labels)} hand trials from {len(np.unique(runs))} recording runs")
    print(f"Model: {args.model} | Chance: 50% | Lab target: 70%")
    if args.tune:  # tuning has its own early and later split
        tune_on_early_runs(trials, labels, runs, args.model)
        return

    settings = [("Baseline", band, electrodes)]  # we always need a starting score
    if args.compare:
        # change only the band or the electrodes from the baseline
        settings.extend([
            ("Mu only", (8, 13), electrodes),
            ("Beta only", (16, 30), electrodes),
            ("Back of head", band, "back"),
        ])
    baseline = None  # fill this in once the first setup has been tested
    for name, test_band, test_channels in settings:
        # a custom baseline might already be the same as one of the changes
        if baseline is not None and (test_band, test_channels) == (band, electrodes):
            print(f"\n{name}: already used in the baseline.")
            continue
        # make the inputs for this setup and predict each held out block
        model, features = prepare_decoder(trials, test_band, test_channels, args.model)
        guesses = held_out_predictions(model, features, labels, runs)
        accuracy = show_result(name, test_band, test_channels, labels, guesses)
        if baseline is None:
            baseline = accuracy
        else:
            # multiply by 100 to show the change in percentage points
            print(f"Difference from baseline: {100 * (accuracy - baseline):+.1f} percentage points")


# replay adds a third option which means stay still
DECISION_TIME = 1.5  # seconds of eeg before making a decision
REST = 2  # zero and one are already used for the two hands
# try allowing guesses freely and then try asking for more certainty
THRESHOLDS = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def make_replay_features(eeg, band, electrodes):
    # replay keeps the original 250 samples per sec
    sections = butter(4, band, btype="bandpass", fs=SAMPLE_RATE, output="sos")
    # go forwards only so the filter cannot see future samples
    signal = sosfilt(sections, pick_channels(eeg, electrodes), axis=-1)
    length = int(DECISION_TIME * SAMPLE_RATE)  # number of samples in one window
    # use before the arrow as a stand in for rest
    rest = log_band_power(signal[:, :, CUE_SAMPLE - length:CUE_SAMPLE])
    # take the same length starting at the arrow for the move
    move = log_band_power(signal[:, :, CUE_SAMPLE:CUE_SAMPLE + length])
    return move, rest


def fit_replay(move, rest, labels):
    features = np.concatenate([move, rest])  # put the rest windows after the moves
    # give every rest row the label two in the same order
    targets = np.concatenate([labels, np.full(len(rest), REST)])
    # fit the same kind of model but now with three answers
    return logistic_model().fit(features, targets)


def commands(probabilities, threshold):
    predicted = probabilities.argmax(axis=1)  # pick left right or rest
    # stay still if even the biggest guess is too low
    # np where picks one answer per row using that confidence check
    return np.where(probabilities.max(axis=1) >= threshold, predicted, REST)


def score_replay(move_prob, rest_prob, labels, threshold):
    # a wrong hand and a missed move both count against correct moves
    correct = int(np.sum(commands(move_prob, threshold) == labels))
    # either hand is a false move when the person was supposed to be resting
    false_alarms = int(np.sum(commands(rest_prob, threshold) != REST))
    return correct, false_alarms


def bits_per_minute(correct_rate):
    # rough two choice speed estimate from the intended moves
    if correct_rate <= 0.5:
        return 0.0  # at chance or below the estimate is zero
    if correct_rate == 1.0:
        return 60 / DECISION_TIME  # a perfect two choice decision gives one bit
    # use the mix of right and wrong choices to estimate bits per decision
    error_rate = 1 - correct_rate
    bits = 1 + correct_rate * log2(correct_rate) + error_rate * log2(error_rate)
    # multiply by how many decisions fit in a minute
    return bits * 60 / DECISION_TIME


def show_replay_score(correct, false_alarms, total, n_electrodes):
    # there is one move window and one rest window for each trial
    correct_rate = correct / total
    false_rate = false_alarms / total
    # each requirement has to pass at the same time
    passed = (
        correct_rate >= 0.70 and false_rate <= 0.20  # enough right with few false moves
        and DECISION_TIME <= 1.5 and n_electrodes <= 13  # fast enough with few enough sensors
    )
    verdict = "PASS" if passed else "FAIL"
    print(f"{correct}/{total} correct ({correct_rate:.1%}) | "
          f"{false_alarms}/{total} false activations ({false_rate:.1%}) | "
          f"{bits_per_minute(correct_rate):.1f} estimated bits/min | {verdict}")
    return passed  # keep only the setups that passed practice


def run_replay(eeg, labels, runs, band, electrodes):
    # get one move window and one rest window from each trial
    move, rest = make_replay_features(eeg, band, electrodes)
    # use the first four blocks to try thresholds and save the rest for later
    early = runs <= 4
    move_early, rest_early, labels_early = move[early], rest[early], labels[early]
    total = len(labels_early)  # the denominator for the practice rates
    n_electrodes = len(CHANNEL_SETS[electrodes])  # part of the lab pass criteria
    # save the three probabilities per move trial and do the same for rest
    move_prob = np.zeros((total, 3))
    rest_prob = np.zeros_like(move_prob)  # same size as the table for moves
    # keep the move and rest from the same trial in the same training or test group
    for train, test in LeaveOneGroupOut().split(move_early, labels_early, runs[early]):
        # train on the other practice blocks before guessing this one
        model = fit_replay(move_early[train], rest_early[train], labels_early[train])
        # store the guesses back at the index for that trial
        move_prob[test] = model.predict_proba(move_early[test])
        rest_prob[test] = model.predict_proba(rest_early[test])

    print(f"Practice on runs 1 to 4 | {DECISION_TIME:.1f} s | {n_electrodes} electrodes")
    passing = []  # save options we can use in the later test
    # use the same guesses for each threshold so only the trigger changes
    for threshold in THRESHOLDS:
        correct, false_alarms = score_replay(move_prob, rest_prob, labels_early, threshold)
        print(f"Threshold {threshold:.2f} | ", end="")
        if show_replay_score(correct, false_alarms, total, n_electrodes):
            # the negative count makes more correct moves win a tie later
            passing.append((false_alarms, -correct, threshold))
    if not passing:  # stop before scoring the later blocks if practice failed
        print("\nNo practice threshold met both targets. Runs 5 and 6 remain unscored.")
        return

    # pick the passing threshold with the fewest false moves and then most correct
    _, _, threshold = min(passing)
    print(f"\nLocked threshold: {threshold:.2f}")
    # fit on all the practice data then score one time with the locked threshold
    model = fit_replay(move_early, rest_early, labels_early)
    later = ~early  # all the trials left out of practice
    correct, false_alarms = score_replay(
        model.predict_proba(move[later]), model.predict_proba(rest[later]), labels[later], threshold
    )
    print("Later recording (runs 5 and 6)")
    show_replay_score(correct, false_alarms, len(labels[later]), n_electrodes)


def main():
    # running this file by itself starts the two electrode model
    parser = argparse.ArgumentParser(description="Decode imagined left and right hand movement from EEG.")
    # choose a model and optionally change the sensors or filter band
    parser.add_argument("--model", choices=("logistic", "csp"), default="logistic")
    parser.add_argument("--electrodes", choices=CHANNEL_SETS)
    # the band option needs two frequencies in hz like 3 and 33
    parser.add_argument("--band", nargs=2, type=float, default=DEFAULT_BAND,
                        metavar=("LOW_HZ", "HIGH_HZ"))
    mode = parser.add_mutually_exclusive_group()  # only one of these modes at a time
    mode.add_argument("--compare", action="store_true", help="Try three one-setting changes.")
    mode.add_argument("--tune", action="store_true", help="Pick settings on early runs and test later.")
    mode.add_argument("--replay", action="store_true", help="Try left, right and stay still.")
    # point to a saved copy of the graz file if you already have one
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    band = tuple(args.band)
    # with 80 samples per sec the offline filter has to stay below 40 hz
    if not 0 < band[0] < band[1] < ANALYSIS_RATE / 2:
        parser.error("Choose an increasing band above 0 Hz and below 40 Hz.")
    # the three answer rest test only uses logistic regression here
    if args.replay and args.model != "logistic":
        parser.error("Replay uses the logistic model.")

    # the basic run uses two electrodes and replay starts with the motor strip
    electrodes = args.electrodes or ("motor" if args.replay else "hands")
    # load the original trials once and pass them to whichever test we asked for
    eeg, labels, runs = load_trials(args.data_dir)
    if args.replay:
        run_replay(eeg, labels, runs, band, electrodes)
    else:
        run_offline(eeg, labels, runs, args, band, electrodes)


# importing the functions somewhere else does not start an experiment
if __name__ == "__main__":
    main()
