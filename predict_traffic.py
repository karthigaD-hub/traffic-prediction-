# ===============================================================
# 🔮 TRAFFIC FORECAST INFERENCE
# ===============================================================
import numpy as np, torch
from traffic_model import TrafficLSTM

DATA_PATH = "data/metr-la/processed.npz"
MODEL_PATH = "traffic_lstm.pt"
SEQ_LEN = 12
HORIZON = 12

data = np.load(DATA_PATH, allow_pickle=True)
X = np.nan_to_num(data["data"]).astype(np.float32)
n_sensors, total_steps = X.shape

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TrafficLSTM(n_sensors, SEQ_LEN, HORIZON).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

last_seq = X[:, -SEQ_LEN:].T  # shape (SEQ_LEN, n_sensors)
input_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(input_tensor).cpu().numpy()[0]

print("✅ Prediction shape:", pred.shape)
print(pred[:3])  # first few predicted steps
np.savetxt("predictions.csv", pred.reshape(HORIZON, n_sensors), delimiter=",")
print("✅ Saved predictions.csv")
