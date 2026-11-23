from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from config import RECORD_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_STARTS


LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
WAVE_LABELS = {"p": 1, "t": 2, "N": 3}
SOURCE_FS = 500


def _wave_labels(symbols: list[str], samples: np.ndarray) -> np.ndarray:
    labels = np.zeros(RECORD_SAMPLES + 2 * SAMPLE_RATE, dtype=np.int8)
    start = None
    wave = None
    previous_wave = None
    previous_offset = None
    fiducial_start = None
    late_qrs_offset = None
    duplicate_onset = False
    expect_p = False

    for index, (symbol, sample) in enumerate(zip(symbols, samples, strict=True)):
        position = round(int(sample) * SAMPLE_RATE / SOURCE_FS)
        if not 0 <= position < len(labels):
            raise ValueError(f"Annotation outside the ECG: {position}")
        before = symbols[index - 1] if index else None
        after = symbols[index + 1] if index + 1 < len(symbols) else None

        if symbol == "(":
            late_qrs_offset = None
            if start is not None:
                if fiducial_start == index - 1:
                    continue
                if wave is None:
                    duplicate_onset = True
                    continue
                if wave != "t":
                    raise ValueError(f"Missing offset for {wave}")
                labels[start:position] = WAVE_LABELS[wave]
                previous_wave = wave
                expect_p = True
            start = position
            wave = None
            previous_offset = None
            fiducial_start = None
            duplicate_onset = False
        elif symbol == ")":
            if start is None:
                if previous_wave == "t" and before == ")" and previous_offset is not None:
                    labels[previous_offset + 1 : position + 1] = WAVE_LABELS["t"]
                    previous_offset = position
                    continue
                raise ValueError("Offset without an active wave")
            if duplicate_onset or expect_p or position < start:
                raise ValueError("Incomplete wave annotation")
            late_qrs_offset = None
            if wave is None:
                between_t_waves = (
                    previous_wave == "t"
                    and index >= 3
                    and symbols[index - 3 : index + 3] == ["t", ")", "(", ")", "(", "t"]
                )
                if before == "(" and after == "N":
                    late_qrs_offset = position
                elif not between_t_waves:
                    raise ValueError("Wave interval without a fiducial")
                wave = "N"
            labels[start : position + 1] = WAVE_LABELS[wave]
            previous_wave, previous_offset = wave, position
            start = None
            wave = None
            fiducial_start = None
        elif symbol in WAVE_LABELS:
            if symbol == "N" and start is None and late_qrs_offset is not None:
                labels[late_qrs_offset + 1 : position + 1] = WAVE_LABELS["N"]
                late_qrs_offset = None
                continue
            if start is None:
                if symbol == "t" or (symbol == "p" and after != "("):
                    raise ValueError(f"Missing onset for {symbol}")
                start = position
                fiducial_start = index
            if duplicate_onset and symbol != "t":
                raise ValueError(f"Repeated onset for {symbol}")
            if expect_p and symbol != "p":
                raise ValueError("Expected a P wave after an open T wave")
            if wave is not None and (wave != symbol or symbol != "t"):
                raise ValueError(f"Unexpected fiducial {symbol}")
            duplicate_onset = False
            expect_p = False
            wave = symbol
        else:
            late_qrs_offset = None

    if start is not None or expect_p:
        raise ValueError("Unclosed wave annotation")
    return labels[SAMPLE_RATE:-SAMPLE_RATE]


def _standardize(signal: np.ndarray) -> np.ndarray:
    mean = float(signal.mean(dtype=np.float64))
    std = max(float(signal.std(dtype=np.float64)), 1e-8)
    return ((signal - mean) / std).astype(np.float32)


def load_ludb(root: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    root = Path(root).expanduser()
    lines = (root / "RECORDS").read_text(encoding="utf-8").splitlines()
    records = [line.strip() for line in lines if line.strip()]
    if len(records) != 200 or len(set(records)) != len(records):
        raise ValueError("LUDB RECORDS must list 200 unique records")

    signals = np.empty(
        (len(records), len(LEADS), len(WINDOW_STARTS), WINDOW_SAMPLES), dtype=np.float32
    )
    labels = np.empty((len(records), len(LEADS), RECORD_SAMPLES), dtype=np.int8)
    bandpass = butter(3, (0.5, 40.0), btype="bandpass", fs=SOURCE_FS, output="sos")

    for record_index, record_id in enumerate(records):
        path = str(root / record_id)
        record = wfdb.rdrecord(path)
        if (
            record.fs != SOURCE_FS
            or record.p_signal.shape != (10 * SOURCE_FS, len(LEADS))
            or set(record.sig_name) != set(LEADS)
        ):
            raise ValueError(f"Unexpected LUDB record format: {record_id}")
        for lead_index, lead in enumerate(LEADS):
            signal = record.p_signal[:, record.sig_name.index(lead)]
            filtered = sosfiltfilt(bandpass, signal)
            resampled = resample_poly(filtered, SAMPLE_RATE, SOURCE_FS)
            central = _standardize(resampled[SAMPLE_RATE:-SAMPLE_RATE])
            for window_index, start in enumerate(WINDOW_STARTS):
                stop = start + WINDOW_SAMPLES
                signals[record_index, lead_index, window_index] = _standardize(central[start:stop])
            annotation = wfdb.rdann(path, lead)
            try:
                labels[record_index, lead_index] = _wave_labels(annotation.symbol, annotation.sample)
            except ValueError as error:
                raise ValueError(f"Invalid LUDB annotation in {record_id}/{lead}: {error}") from error

    return records, signals, labels
