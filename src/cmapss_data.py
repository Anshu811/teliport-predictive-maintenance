"""Loading utilities for the NASA C-MAPSS turbofan dataset."""
from pathlib import Path
import pandas as pd

ID_COLS = ["unit", "cycle"]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ID_COLS + OP_COLS + SENSOR_COLS
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SUBSET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1},
    "FD002": {"conditions": 6, "fault_modes": 1},
    "FD003": {"conditions": 1, "fault_modes": 2},
    "FD004": {"conditions": 6, "fault_modes": 2},
}


def _read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS
    df[ID_COLS] = df[ID_COLS].astype(int)
    return df


def load_subset(fd: str, data_dir):
    """Return (train, test, rul_true) for one subset.

    train : run-to-failure rows with true RUL label (uncapped)
    test  : truncated trajectories (RUL unknown at row level)
    rul_true : one row per test unit -> true RUL at its last cycle
    """
    data_dir = Path(data_dir)
    train = _read_txt(data_dir / f"train_{fd}.txt")
    test = _read_txt(data_dir / f"test_{fd}.txt")

    rul = pd.read_csv(data_dir / f"RUL_{fd}.txt", sep=r"\s+", header=None).iloc[:, :1]
    rul.columns = ["RUL"]
    rul.insert(0, "unit", range(1, len(rul) + 1))

    units = sorted(test["unit"].unique())
    assert units == list(range(1, len(units) + 1)), "Test units not contiguous"
    assert len(rul) == len(units), "RUL file does not match test units"

    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    for df in (train, test, rul):
        df["subset"] = fd
    return train, test, rul


def load_all(data_dir, subsets=SUBSETS):
    return {fd: load_subset(fd, data_dir) for fd in subsets}
