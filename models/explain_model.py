import numpy as np
import torch
import torch.nn.functional as F

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
from train_model import CNN_GAT

model = CNN_GAT().to(device)
model.load_state_dict(torch.load("models/cnn_gat_model.pth", map_location=device))
model.eval()

# =========================
# LOAD DATA
# =========================
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

delta = np.load("data/delta_test.npy")
theta = np.load("data/theta_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

# =========================
# ELECTRODE NAMES
# =========================
electrodes = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T3','T4','T5','T6','Cz'
]

# =========================
# PICK SAMPLE (CONTROLLED)
# =========================
mode = "abnormal"  # change to "normal" or "abnormal"

if mode == "abnormal":
    indices = np.where(y_test == 1)[0]
else:
    indices = np.where(y_test == 0)[0]

if len(indices) == 0:
    raise ValueError("No samples found for selected mode!")
# random sampling
i = np.random.choice(indices)
sample = X_test[i].unsqueeze(0)
# for multiple demo
# for i in indices[:5]:
#     sample = X_test[i].unsqueeze(0)

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    output = model(sample)
    probs = F.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)

prediction = "Abnormal" if pred.item() == 1 else "Normal"
confidence = conf.item() * 100

# =========================
# IMPORTANT CHANNELS (BETTER LOGIC)
# =========================
# use variance instead of mean (more meaningful for EEG)
channel_activity = np.var(sample.cpu().numpy(), axis=2)[0]
top_indices = np.argsort(channel_activity)[-3:]

important_channels = [electrodes[i] for i in top_indices]

# =========================
# DELTA / THETA ANALYSIS
# =========================
delta_val = np.mean(delta[i])
theta_val = np.mean(theta[i])

# safer baseline (avoid leakage bias)
normal_indices = np.where(y_test == 0)[0]

delta_mean_normal = np.mean(delta[normal_indices])
theta_mean_normal = np.mean(theta[normal_indices])

delta_status = "High" if delta_val > delta_mean_normal else "Normal"
theta_status = "High" if theta_val > theta_mean_normal else "Normal"

# =========================
# FINAL OUTPUT
# =========================
print("\n========================")
print("EEG ANALYSIS RESULT")
print("========================")

print("Prediction:", prediction)
print(f"Confidence: {confidence:.2f}%")

print("\nImportant Electrodes:", important_channels)

print("\nSlow Wave Analysis:")
print("Delta:", delta_status)
print("Theta:", theta_status)

if prediction == "Abnormal":
    if delta_status == "High" or theta_status == "High":
        print("\nConclusion: Slow-wave abnormality detected (possible encephalopathy)")
    else:
        print("\nConclusion: Abnormal EEG pattern detected (non slow-wave)")
else:
    print("\nConclusion: Normal EEG activity")