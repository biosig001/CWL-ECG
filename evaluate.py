import numpy as np
import torch
from scipy.special import softmax

from config import BATCH_SIZE, RECORD_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_STARTS


WAVES = {"P": 1, "QRS": 3, "T": 2}
TOLERANCE = 70 * SAMPLE_RATE / 1000


def _intervals(mask):
    changes = np.diff(np.pad(mask.astype(np.int8), 1))
    return np.column_stack((np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def _lead_annotations(labels):
    intervals = {wave: _intervals(labels == code) for wave, code in WAVES.items()}
    qrs = intervals["QRS"]
    qrs = qrs[(qrs[:, 0] > 0) & (qrs[:, 1] < RECORD_SAMPLES - 1)]
    return intervals, qrs, qrs.mean(axis=1)


def fit_gates(training_labels: np.ndarray) -> dict[str, bool]:
    dispersions = {wave: [] for wave in WAVES}
    for record in training_labels:
        leads = [_lead_annotations(labels) for labels in record]
        lengths = np.array([len(centers) for _, _, centers in leads])
        counts = np.bincount(lengths)
        common_length = np.flatnonzero(counts == counts.max())[-1]
        reference = leads[np.flatnonzero(lengths == common_length)[0]][2]
        for center in reference:
            events = {wave: [] for wave in WAVES}
            for intervals, qrs, centers in leads:
                if not len(centers):
                    continue
                index = int(np.argmin(np.abs(centers - center)))
                anchor = centers[index]
                if abs(anchor - center) > 16:
                    continue
                onset, offset = qrs[index]
                events["QRS"].append(qrs[index] - anchor)
                left = centers[index - 1] if index else -1
                right = centers[index + 1] if index + 1 < len(centers) else np.inf
                p_wave = intervals["P"]
                t_wave = intervals["T"]
                preceding = p_wave[(p_wave[:, 0] > left) & (p_wave[:, 1] < onset)]
                following = t_wave[(t_wave[:, 0] > offset) & (t_wave[:, 1] < right)]
                if len(preceding):
                    events["P"].append(preceding[-1] - anchor)
                if len(following):
                    events["T"].append(following[0] - anchor)
            for wave, times in events.items():
                if len(times) >= 7:
                    values = np.asarray(times)
                    deviations = np.abs(values - np.median(values, axis=0))
                    dispersions[wave].append(1.4826 * np.median(deviations, axis=0))
    return {
        wave: bool(values and np.all(np.median(values, axis=0) <= 2))
        for wave, values in dispersions.items()
    }


@torch.inference_mode()
def predict_logits(model, signals: np.ndarray, device) -> np.ndarray:
    model.eval()
    windows = torch.from_numpy(np.ascontiguousarray(signals)).reshape(-1, 1, WINDOW_SAMPLES)
    logits = np.empty((len(windows), 4, WINDOW_SAMPLES), dtype=np.float32)
    for start in range(0, len(windows), BATCH_SIZE):
        batch = windows[start : start + BATCH_SIZE].to(device)
        logits[start : start + len(batch)] = model(batch).cpu().numpy()
    return logits.reshape(*signals.shape[:-1], 4, WINDOW_SAMPLES)


def _fuse_leads(logits):
    background = logits[..., :1, :]
    foreground = logits[..., 1:, :] - background
    median = np.median(foreground, axis=1, keepdims=True)
    deviation = foreground - median
    mad = np.median(np.abs(deviation), axis=1, keepdims=True)
    consensus_variance = np.pi * np.square(1.482602218505602 * mad) / (2 * logits.shape[1])
    local_error = np.square(deviation)
    denominator = local_error + consensus_variance
    alpha = np.divide(local_error, denominator, out=np.zeros_like(local_error), where=denominator > 0)
    fused = logits.copy()
    fused[..., 1:, :] = background + foreground - alpha * deviation
    return fused


def _reconstruct(windows):
    logits = np.zeros((*windows.shape[:2], 4, RECORD_SAMPLES), dtype=windows.dtype)
    counts = np.zeros(RECORD_SAMPLES, dtype=np.int8)
    for window, start in enumerate(WINDOW_STARTS):
        stop = start + WINDOW_SAMPLES
        logits[..., start:stop] += windows[:, :, window]
        counts[start:stop] += 1
    return logits / counts


def _decode(logits):
    probabilities = softmax(logits.reshape(-1, 4, RECORD_SAMPLES), axis=1)
    labels = probabilities.argmax(axis=1)
    alternative_codes = np.array([0, WAVES["T"], WAVES["QRS"]])
    p_code = WAVES["P"]
    for probability, prediction in zip(probabilities, labels):
        p = probability[p_code]
        eligible = p >= np.maximum(probability[WAVES["T"]], probability[WAVES["QRS"]])
        accepted = np.zeros(RECORD_SAMPLES, dtype=bool)
        for start, end in _intervals(eligible & (p >= 0.20)):
            if end - start + 1 >= 5 and p[start : end + 1].max() >= 0.35:
                accepted[start : end + 1] = True
        rejected = (prediction == p_code) & ~accepted
        alternatives = probability[alternative_codes][:, rejected]
        prediction[rejected] = alternative_codes[alternatives.argmax(axis=0)]
        prediction[accepted & (prediction == 0)] = p_code
    return labels


def _match_count(reference, prediction):
    i = j = matches = 0
    while i < len(reference) and j < len(prediction):
        difference = prediction[j] - reference[i]
        if difference < -TOLERANCE:
            j += 1
        elif difference > TOLERANCE:
            i += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def _percent(numerator, denominator):
    return float(100 * numerator / denominator) if denominator else 0.0


def _scores(tp, fp, fn, precision_name, recall_name):
    return {
        precision_name: _percent(tp, tp + fp),
        recall_name: _percent(tp, tp + fn),
        "f1": _percent(2 * tp, 2 * tp + fp + fn),
    }


def _add_average(metrics):
    rows = tuple(metrics.values())
    metrics["Average"] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate(logits: np.ndarray, labels: np.ndarray, gates: dict[str, bool]) -> dict:
    truth = labels.reshape(-1, RECORD_SAMPLES)
    local = _decode(_reconstruct(logits))
    fused = _decode(_reconstruct(_fuse_leads(logits)))
    classification = {}
    boundaries = {}
    for wave, code in WAVES.items():
        actual = truth == code
        predicted = local == code
        tp = np.count_nonzero(actual & predicted)
        fp = np.count_nonzero(predicted) - tp
        fn = np.count_nonzero(actual) - tp
        classification[wave] = _scores(tp, fp, fn, "precision", "recall")

        predictions = fused if gates[wave] else local
        references = [_intervals(mask) for mask in actual]
        intervals = [_intervals(sequence == code) for sequence in predictions]
        n_references = sum(map(len, references))
        n_predictions = sum(map(len, intervals))
        for side, name in enumerate(("onset", "offset")):
            tp = sum(_match_count(ref[:, side], pred[:, side]) for ref, pred in zip(references, intervals))
            boundaries[f"{wave} {name}"] = _scores(
                tp, n_predictions - tp, n_references - tp, "ppv", "sensitivity"
            )
    _add_average(classification)
    _add_average(boundaries)
    return {"classification": classification, "boundaries": boundaries}


def summarize(folds: list[dict]) -> dict:
    summary = {}
    for group, rows in folds[0].items():
        summary[group] = {}
        for name, metrics in rows.items():
            summary[group][name] = {}
            for metric in metrics:
                values = [fold[group][name][metric] for fold in folds]
                summary[group][name][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary
