# kdd_to_cicids_converter.py could scrap

import pandas as pd
import numpy as np

# -----------------------------
# Load KDD dataset
# -----------------------------
def load_kdd(file_path):
    columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","label"
    ]

    df = pd.read_csv(file_path, names=columns)
    return df


# -----------------------------
# Convert to CICIDS-like format
# -----------------------------
def convert_to_cicids(df):
    new_df = pd.DataFrame()

    # Basic mappings
    new_df["Flow Duration"] = df["duration"]
    new_df["Total Fwd Packets"] = df["count"]
    new_df["Total Backward Packets"] = df["srv_count"]

    new_df["Total Length of Fwd Packets"] = df["src_bytes"]
    new_df["Total Length of Bwd Packets"] = df["dst_bytes"]

    # Approximate rates
    new_df["Flow Bytes/s"] = (
        (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1)
    )

    new_df["Flow Packets/s"] = (
        (df["count"] + df["srv_count"]) / (df["duration"] + 1)
    )

    # Flags (rough approximation)
    new_df["SYN Flag Count"] = df["serror_rate"] * 100
    new_df["RST Flag Count"] = df["rerror_rate"] * 100

    # Binary features
    new_df["PSH Flag Count"] = df["urgent"]
    new_df["ACK Flag Count"] = df["logged_in"]

    # Placeholder features (not present in KDD)
    missing_features = [
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Packet Length Std"
    ]

    for col in missing_features:
        new_df[col] = 0

    # -----------------------------
    # Label Mapping
    # -----------------------------
    def map_label(label):
        if label == "normal.":
            return "BENIGN"
        else:
            return "ATTACK"

    new_df["Label"] = df["label"].apply(map_label)

    return new_df


# -----------------------------
# Save Output
# -----------------------------
def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved converted dataset to {output_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    input_file = "data\kddcup.data"
    output_file = "data\kdd_converted_cicids.csv"

    kdd_df = load_kdd(input_file)
    cicids_df = convert_to_cicids(kdd_df)
    save_dataset(cicids_df, output_file)