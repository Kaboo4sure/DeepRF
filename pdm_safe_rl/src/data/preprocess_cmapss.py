#<
#This produces train/test tables with:

#unit, cycle, settings + sensors

#RUL label for training units

import os
import numpy as np
import pandas as pd

COLS = (["unit", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"s{i}" for i in range(1, 22)])

def load_fd(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    return df

def compute_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    # RUL = max_cycle(unit) - cycle
    max_cycle = df_train.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df_train.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df

def main(dataset="FD001"):
    raw_dir = "data/raw/cmapss/CMAPSSData"
    out_dir = "data/processed/cmapss"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, f"train_{dataset}.txt")
    test_path  = os.path.join(raw_dir, f"test_{dataset}.txt")
    rul_path   = os.path.join(raw_dir, f"RUL_{dataset}.txt")

    train_df = load_fd(train_path)
    test_df  = load_fd(test_path)

    # True RUL for each test unit is provided as a single value per unit (at the end of test)
    rul_end = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].values

    train_df = compute_rul(train_df)

    # For test: compute RUL per row using end-of-life RUL + (max_cycle - cycle)
    max_cycle_test = test_df.groupby("unit")["cycle"].max().sort_index().values
    # RUL at each cycle = RUL_end(unit) + (max_cycle(unit) - cycle)
    test_df = test_df.copy()
    test_df["max_cycle"] = test_df.groupby("unit")["cycle"].transform("max")
    # units in FD are 1..N
    unit_ids = test_df["unit"].unique()
    unit_to_rul_end = {u: rul_end[i] for i, u in enumerate(unit_ids)}
    test_df["RUL_end"] = test_df["unit"].map(unit_to_rul_end)
    test_df["RUL"] = test_df["RUL_end"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df.drop(columns=["max_cycle", "RUL_end"], inplace=True)

    train_out = os.path.join(out_dir, f"{dataset}_train.csv")
    test_out  = os.path.join(out_dir, f"{dataset}_test.csv")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print("Saved:")
    print(train_out)
    print(test_out)
    print(train_df.head())

if __name__ == "__main__":
    main(dataset="FD001")
