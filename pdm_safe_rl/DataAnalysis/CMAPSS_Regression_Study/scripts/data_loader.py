"""Load and prepare NASA C-MAPSS training data for RUL analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project and dataset paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CMAPSS_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "cmapss"
    / "CMAPSSData"
)


# ---------------------------------------------------------------------
# C-MAPSS column names
# ---------------------------------------------------------------------

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def load_cmapss_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS text file and assign column names.

    Parameters
    ----------
    file_path:
        Path to a C-MAPSS file such as train_FD001.txt.

    Returns
    -------
    pd.DataFrame
        Data containing unit, cycle, operating settings,
        and sensor measurements.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file was not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"The supplied path is not a file:\n{path}"
        )

    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    expected_columns = len(COLUMN_NAMES)

    if data.shape[1] < expected_columns:
        raise ValueError(
            f"Expected at least {expected_columns} columns, "
            f"but found {data.shape[1]}."
        )

    data = data.iloc[:, :expected_columns].copy()
    data.columns = COLUMN_NAMES

    return data


def add_rul(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life for each engine cycle.

    RUL = maximum cycle for the engine - current cycle
    """
    required_columns = {"unit", "cycle"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    result = data.copy()

    maximum_cycles = (
        result.groupby("unit")["cycle"]
        .max()
        .rename("max_cycle")
    )

    result = result.merge(
        maximum_cycles,
        on="unit",
        how="left",
        validate="many_to_one",
    )

    result["RUL"] = (
        result["max_cycle"] - result["cycle"]
    )

    result = result.drop(columns=["max_cycle"])

    return result


def validate_data(data: pd.DataFrame) -> None:
    """Run basic checks on the prepared C-MAPSS dataset."""
    if data.empty:
        raise ValueError("The loaded dataset is empty.")

    missing_values = int(data.isna().sum().sum())

    if missing_values > 0:
        raise ValueError(
            f"The dataset contains {missing_values} missing values."
        )

    if "RUL" in data.columns and (data["RUL"] < 0).any():
        raise ValueError("Negative RUL values were detected.")

    duplicate_rows = int(data.duplicated().sum())

    print("\nData validation")
    print("----------------")
    print(f"Rows: {data.shape[0]:,}")
    print(f"Columns: {data.shape[1]}")
    print(f"Engines: {data['unit'].nunique()}")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate rows: {duplicate_rows}")

    if "RUL" in data.columns:
        print(
            f"RUL range: "
            f"{data['RUL'].min()} to "
            f"{data['RUL'].max()} cycles"
        )


def load_training_data(
    dataset: str = "FD001",
) -> pd.DataFrame:
    """
    Load a C-MAPSS training dataset and calculate RUL.

    Parameters
    ----------
    dataset:
        Dataset identifier. Valid options are FD001, FD002,
        FD003, and FD004.

    Returns
    -------
    pd.DataFrame
        Prepared C-MAPSS training data with an RUL column.
    """
    dataset = dataset.upper()

    valid_datasets = {
        "FD001",
        "FD002",
        "FD003",
        "FD004",
    }

    if dataset not in valid_datasets:
        raise ValueError(
            f"Invalid dataset '{dataset}'. "
            f"Choose from {sorted(valid_datasets)}."
        )

    file_path = CMAPSS_DIR / f"train_{dataset}.txt"

    data = load_cmapss_file(file_path)
    data = add_rul(data)
    validate_data(data)

    return data


def main() -> None:
    """Load FD001 and display a basic summary."""
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nC-MAPSS directory:")
    print(CMAPSS_DIR)

    data = load_training_data("FD001")

    print("\nFirst five rows")
    print("----------------")
    print(data.head())

    print("\nDataset shape")
    print("-------------")
    print(data.shape)

    print("\nRUL summary")
    print("-----------")
    print(data["RUL"].describe())


if __name__ == "__main__":
    main()