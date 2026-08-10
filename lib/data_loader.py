# load_datasets_new_format
# load_data
# load_features

"""
data_loader.py
==============
EMG data loading functions.
"""

import os
import pandas as pd
import pickle as pkl
import joblib as jb


# ─────────────────────────────────────────────────────────────────────────────
# New format: YYYYMMDD_HHMMSS_emg.csv
# Expected columns: Time(ms), Raw_CH0..3, Amp_CH0..3, Label, Window
# ─────────────────────────────────────────────────────────────────────────────



def load_datasets_new_format(
    data_folder: str,
    actions: dict,
    labels: list
) -> list[pd.DataFrame]:
    """
    Load all CSV files from a folder
    (format YYYYMMDD_HHMMSS_emg.csv).

    Each file is treated as an independent take.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the CSV files.

    actions : dict
        Dictionary {action_name: integer_code}.
        Labels missing from the dictionary receive code -1.

    labels : list[str]
        Raw signal column names
        (e.g. ['Raw_CH0', ..., 'Raw_CH3']).

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, one per CSV file.
    """

    csv_files = sorted([
        f for f in os.listdir(data_folder)
        if f.endswith(".csv")
    ])

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_folder}"
        )

    all_df = []

    for fname in csv_files:

        fpath = os.path.join(data_folder, fname)

        try:
            # Check if the file is empty (0 bytes)
            if os.path.getsize(fpath) == 0:
                print(f"  ⚠️ Empty file ignored: {fname}")
                continue

            # Read CSV
            df = pd.read_csv(fpath)

            # Check if the DataFrame is empty
            if df.empty or len(df.columns) == 0:
                print(f"  ⚠️ Invalid CSV ignored: {fname}")
                continue

        except pd.errors.EmptyDataError:
            print(f"  ⚠️ EmptyDataError ignored: {fname}")
            continue

        except Exception as e:
            print(f"  ❌ Error reading {fname}: {e}")
            continue

        # -----------------------------
        # Rename time column
        # -----------------------------
        if "Time(ms)" in df.columns:
            df.rename(
                columns={"Time(ms)": "Timestamp"},
                inplace=True
            )

            # ms -> seconds
            df["Timestamp"] = df["Timestamp"] / 1000.0

        # -----------------------------
        # EMG column validation
        # -----------------------------
        missing_labels = [
            col for col in labels
            if col not in df.columns
        ]

        if missing_labels:
            print(
                f"  ⚠️ Missing columns in {fname}: "
                f"{missing_labels}"
            )

        # -----------------------------
        # Label mapping
        # -----------------------------
        if "Label" in df.columns:

            df["Action"] = (
                df["Label"]
                .map(actions)
                .fillna(-1)
                .astype(int)
            )

        else:
            df["Action"] = -1

        # -----------------------------
        # Save take
        # -----------------------------
        all_df.append(df)

        print(
            f"  ✅ Loaded: {fname} "
            f"({len(df)} rows)"
        )

    print(
        f"\n✔ {len(all_df)} file(s) loaded "
        f"from {data_folder}"
    )

    return all_df

def load_data(filepath: str, actions: dict, labels: list) -> pd.DataFrame:
    """
    Load a single CSV file (new format).

    Parameters
    ----------
    filepath : str
        Full path to the CSV file.
    actions : dict
        Dictionary {action_name: integer_code}.
    labels : list[str]
        Raw signal columns.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)

    if 'Time(ms)' in df.columns:
        df.rename(columns={'Time(ms)': 'Timestamp'}, inplace=True)
        df['Timestamp'] = df['Timestamp'] / 1000.0

    if 'Label' in df.columns:
        df['Action'] = df['Label'].map(actions).fillna(-1).astype(int)
    else:
        df['Action'] = -1

    return df


def load_features(pkl_filename: str, folder_name: str = "results") -> tuple:
    """
    Load a saved feature file in pickle format.

    Parameters
    ----------
    pkl_filename : str
        File name (without extension).
    folder_name : str
        Folder relative to the current directory.

    Returns
    -------
    tuple : (emg_train_z, emg_test_z, actions_train, actions_test)
    """
    features = load_my_model(pkl_filename, folder_name=folder_name, save_type='pkl')
    return (
        features["emg_train"],
        features["emg_test"],
        features["actions_train"],
        features["actions_test"],
    )


def save_features(features: dict, filename: str, folder_name: str = "data/features") -> None:
    """
    Save a feature dictionary into a pickle file.

    Parameters
    ----------
    features : dict
        Dictionary containing feature arrays
        (typical keys: 'emg_train', 'emg_test', 'actions_train', 'actions_test').
    filename : str
        File name (without extension).
    folder_name : str
        Destination folder.
    """
    from lib.export import save_my_model  # local import to avoid circular dependency
    save_my_model(features, filename, folder_name=folder_name)
    print(f"Features saved → {folder_name}/{filename}.pkl")


# ─── Internal alias used in load_features ────────────────────────────────────
def load_my_model(model_name: str, folder_name: str = "results", save_type: str = 'pkl'):
    """Load an object from a pkl or joblib file."""
    model_dir = os.path.join(os.getcwd(), folder_name)
    model_file_path = os.path.join(model_dir, f"{model_name}.{save_type}")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"File not found: {model_file_path}")
    with open(model_file_path, 'rb') as f:
        if save_type == 'pkl':
            return pkl.load(f)
        elif save_type == 'joblib':
            return jb.load(f)
    raise ValueError(f"Unknown save_type: {save_type}")