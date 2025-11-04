import numpy as np
import pandas as pd
import os

DATA_DIR = "data/metr-la"
npy_path = os.path.join(DATA_DIR, "metr_la.npy")
csv_path = os.path.join(DATA_DIR, "metr_la_normalized.csv")
out_path = os.path.join(DATA_DIR, "processed.npz")

if os.path.exists(npy_path):
    print("📊 Loading from NumPy array...")
    data = np.load(npy_path)
elif os.path.exists(csv_path):
    print("📊 Loading from CSV...")
    data = pd.read_csv(csv_path, index_col=0).values
else:
    raise FileNotFoundError("❌ No metr_la.npy or metr_la_normalized.csv found. Run preprocess_from_zip.py first!")

# Transpose if needed (make shape = sensors × timesteps)
if data.shape[0] < data.shape[1]:
    print("↩️ Transposing data to (n_sensors, timesteps)...")
    data = data.T

np.savez_compressed(out_path, data=data)
print(f"✅ Saved {out_path} — shape={data.shape}")
