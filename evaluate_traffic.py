# ===============================================================
# 📊 Evaluate Traffic LSTM + Visualization
# ===============================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from traffic_model import TrafficLSTM
import random

# -----------------------------
# 🔧 Paths and Parameters
# -----------------------------
DATA_PATH = "data/metr-la/processed.npz"
MODEL_PATH = "traffic_lstm.pt"
SEQ_LEN = 12
HORIZON = 12

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 🧩 Load Data
# -----------------------------
data = np.load(DATA_PATH, allow_pickle=True)
X = np.nan_to_num(data["data"].astype(np.float32))

n_sensors, total_steps = X.shape
samples = total_steps - SEQ_LEN - HORIZON

seqs = np.stack([X[:, t:t+SEQ_LEN].T for t in range(samples)])
targets = np.stack([X[:, t+SEQ_LEN:t+SEQ_LEN+HORIZON].T for t in range(samples)])

# Convert to tensors
X_tensor = torch.tensor(seqs).to(device)
y_true = targets

# -----------------------------
# 🧠 Load Trained Model
# -----------------------------
model = TrafficLSTM(n_sensors, SEQ_LEN, HORIZON).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# 🔮 Make Predictions
# -----------------------------
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy()

# -----------------------------
# 📈 Compute Metrics
# -----------------------------
true = y_true.reshape(-1)
pred = preds.reshape(-1)

rmse = mean_squared_error(true, pred) ** 0.5
mae = mean_absolute_error(true, pred)
r2 = r2_score(true, pred)

print("\n📊 Evaluation Results:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE : {mae:.4f}")
print(f"   R²  : {r2:.4f}")

# -----------------------------
# 🎨 Visualization (3 random sensors)
# -----------------------------
plt.figure(figsize=(14, 8))
sensor_ids = random.sample(range(n_sensors), 3)
t_range = np.arange(HORIZON)

for i, sensor in enumerate(sensor_ids):
    plt.subplot(3, 1, i + 1)
    plt.plot(t_range, y_true[-1, :, sensor], label="True", linewidth=2)
    plt.plot(t_range, preds[-1, :, sensor], label="Predicted", linestyle="--")
    plt.title(f"Sensor {sensor}")
    plt.xlabel("Time Steps (5-min intervals)")
    plt.ylabel("Traffic Speed")
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
