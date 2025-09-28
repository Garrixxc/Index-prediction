# src/utils/cv.py
import numpy as np

def rolling_windows(n_samples: int, n_splits: int, test_size: int, embargo: int,
                    min_train: int | None = None):
    """
    Expanding-train / fixed-test time-series splits with embargo.
    Always in-bounds, even on short histories.
    """
    if n_samples <= 0: return
    test_size = max(8, int(test_size))
    embargo   = max(0, int(embargo))
    n_splits  = max(1, int(n_splits))

    if min_train is None:
        min_train = max(104, n_samples // 5)  # ~2y of weeks or 20% of data
    min_train = min(min_train, max(1, n_samples - test_size - embargo))

    last_start = n_samples - test_size
    if last_start <= min_train:  # too little data → single split
        te_start = max(0, last_start)
        tr_end   = max(0, te_start - embargo)
        yield np.arange(0, tr_end), np.arange(te_start, min(te_start + test_size, n_samples))
        return

    # Evenly spaced test-window starts
    starts = np.linspace(min_train, last_start, num=n_splits, dtype=int)
    seen = set()
    for s in starts:
        if s in seen: continue
        seen.add(s)
        tr_end   = max(0, s - embargo)
        te_start = s
        te_end   = min(te_start + test_size, n_samples)
        if tr_end <= 0 or te_end - te_start <= 0:  # skip degenerate windows
            continue
        yield np.arange(0, tr_end), np.arange(te_start, te_end)
