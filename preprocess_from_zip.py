import os
import zipfile
import argparse
import pickle
import numpy as np
import pandas as pd
import h5py

# =========================================================
# 1️⃣ Extract ZIP file
# =========================================================
def extract_zip(zip_path, out_dir):
    print(f"📦 Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("✅ Extracted. Files:", os.listdir(out_dir))


# =========================================================
# 2️⃣ Load HDF5 file robustly
# =========================================================
def load_h5_auto(h5_path):
    print(f"🔍 Reading {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print("📂 H5 keys:", keys)

        if 'df' in keys:
            group = f['df']
            print("📁 Found 'df' group — exploring structure...")

            # Option 1: pickled DataFrame
            try:
                raw = group[()]
                df = pickle.loads(raw)
                print("✅ Loaded pickled DataFrame:", df.shape)
                return df
            except Exception:
                pass

            # Option 2: pandas-style HDF structure (common for METR-LA)
            try:
                if 'block0_values' in group:
                    data = group['block0_values'][:]
                    idx_raw = group['axis0'][:]
                    col_raw = group['axis1'][:]

                    # Decode labels safely
                    index = [str(i, 'utf-8') if isinstance(i, (bytes, np.bytes_)) else str(i)
                             for i in idx_raw]
                    columns = [str(c, 'utf-8') if isinstance(c, (bytes, np.bytes_)) else str(c)
                               for c in col_raw]

                    # Handle shape mismatch (transpose if needed)
                    if data.shape[0] != len(index) and data.shape[1] == len(index):
                        data = data.T
                    if data.shape[1] != len(columns) and data.shape[0] == len(columns):
                        data = data.T

                    df = pd.DataFrame(data, index=index, columns=columns)
                    print("✅ Loaded via pandas-style table:", df.shape)
                    return df
            except Exception as e:
                print("⚠️ Error interpreting as pandas table:", e)

        # Option 3: fallback for simple datasets
        for k in ['speed', 'data', 'values']:
            if k in f:
                print(f"📈 Found dataset '{k}' — reading as array.")
                arr = np.array(f[k])
                df = pd.DataFrame(arr)
                print("✅ Loaded simple dataset:", df.shape)
                return df

    raise ValueError("❌ Could not interpret H5 structure automatically.")


# =========================================================
# 3️⃣ Normalize and Save Data
# =========================================================
def normalize_and_save(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Save raw
    raw_path = os.path.join(out_dir, "metr_la_raw.csv")
    df.to_csv(raw_path)
    print(f"💾 Saved raw data to {raw_path}")

    # Normalize
    norm_df = (df - df.mean()) / (df.std() + 1e-6)
    norm_path = os.path.join(out_dir, "metr_la_normalized.csv")
    norm_df.to_csv(norm_path)
    print(f"💾 Saved normalized data to {norm_path}")

    # NumPy version
    npy_path = os.path.join(out_dir, "metr_la.npy")
    np.save(npy_path, norm_df.values)
    print(f"💾 Saved NumPy array to {npy_path}")


# =========================================================
# 4️⃣ Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess METR-LA traffic dataset")
    parser.add_argument("--zip", required=True, help="Path to dataset zip file")
    parser.add_argument("--out_dir", default="data/metr-la", help="Output directory")
    args = parser.parse_args()

    extract_zip(args.zip, args.out_dir)
    h5_path = os.path.join(args.out_dir, "METR-LA.h5")

    df = load_h5_auto(h5_path)
    normalize_and_save(df, args.out_dir)

    print("🎉 Preprocessing complete!")
