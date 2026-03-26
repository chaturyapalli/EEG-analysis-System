import numpy as np
import os

WINDOW_SIZE = 640

# =========================
# SEGMENT FUNCTION
# =========================
def segment_recording(recording):
    segments = []

    if recording.shape[0] > recording.shape[1]:
        recording = recording.T

    total_samples = recording.shape[1]
    num_segments = total_samples // WINDOW_SIZE

    for i in range(num_segments):
        seg = recording[:, i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]
        if seg.shape[1] == WINDOW_SIZE:
            segments.append(seg)

    return segments

# =========================
# PROCESS DATASET
# =========================
def process_set(dataset):
    all_segments = []

    for rec in dataset:
        rec = np.array(rec)
        segs = segment_recording(rec)
        all_segments.extend(segs)

    return all_segments

# =========================
# LOAD DATA
# =========================
print("Loading recordings...")

openneuro = np.load("data/X_openneuro.npy", allow_pickle=True)
chbmit = np.load("data/X_chbmit.npy", allow_pickle=True)

# =========================
# SPLIT RECORDINGS
# =========================
split_ratio = 0.8

n_open = int(len(openneuro) * split_ratio)
n_chb = int(len(chbmit) * split_ratio)

open_train = openneuro[:n_open]
open_test = openneuro[n_open:]

chb_train = chbmit[:n_chb]
chb_test = chbmit[n_chb:]

print("Segmenting...")

# =========================
# TRAIN DATA (MIXED LOGIC)
# =========================
train_normal = process_set(open_train)

chb_normal = []
chb_abnormal = []

for rec in chb_train:
    rec = np.array(rec)
    mid = rec.shape[1] // 2

    normal_part = rec[:, :mid]
    abnormal_part = rec[:, mid:]

    chb_normal.extend(segment_recording(normal_part))
    chb_abnormal.extend(segment_recording(abnormal_part))

train_normal.extend(chb_normal)
train_abnormal = chb_abnormal

# =========================
# TEST DATA (MIXED LOGIC)
# =========================
test_normal = process_set(open_test)

chb_normal_test = []
chb_abnormal_test = []

for rec in chb_test:
    rec = np.array(rec)
    mid = rec.shape[1] // 2

    normal_part = rec[:, :mid]
    abnormal_part = rec[:, mid:]

    chb_normal_test.extend(segment_recording(normal_part))
    chb_abnormal_test.extend(segment_recording(abnormal_part))

test_normal.extend(chb_normal_test)
test_abnormal = chb_abnormal_test

# =========================
# GLOBAL CHANNEL FIX
# =========================
all_segments = train_normal + train_abnormal + test_normal + test_abnormal
min_channels = min(seg.shape[0] for seg in all_segments)

def fix_channels(segments):
    return [seg[:min_channels, :] for seg in segments]

train_normal = fix_channels(train_normal)
train_abnormal = fix_channels(train_abnormal)
test_normal = fix_channels(test_normal)
test_abnormal = fix_channels(test_abnormal)

# Convert to numpy
train_normal = np.array(train_normal, dtype=np.float32)
train_abnormal = np.array(train_abnormal, dtype=np.float32)
test_normal = np.array(test_normal, dtype=np.float32)
test_abnormal = np.array(test_abnormal, dtype=np.float32)

print("Channels used:", min_channels)

# =========================
# BALANCE TRAIN
# =========================
min_train = min(len(train_normal), len(train_abnormal))
train_normal = train_normal[:min_train]
train_abnormal = train_abnormal[:min_train]

# =========================
# BALANCE TEST
# =========================
min_test = min(len(test_normal), len(test_abnormal))
test_normal = test_normal[:min_test]
test_abnormal = test_abnormal[:min_test]

# =========================
# FINAL DATA
# =========================
X_train = np.concatenate((train_normal, train_abnormal))
y_train = np.concatenate((
    np.zeros(len(train_normal)),
    np.ones(len(train_abnormal))
))

X_test = np.concatenate((test_normal, test_abnormal))
y_test = np.concatenate((
    np.zeros(len(test_normal)),
    np.ones(len(test_abnormal))
))

# Shuffle
train_idx = np.random.permutation(len(X_train))
test_idx = np.random.permutation(len(X_test))

X_train = X_train[train_idx]
y_train = y_train[train_idx]

X_test = X_test[test_idx]
y_test = y_test[test_idx]

# =========================
# SAVE
# =========================
os.makedirs("data", exist_ok=True)

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("\nDONE")
print("Train:", X_train.shape)
print("Test:", X_test.shape)


# =========================
# 🔥 DELTA / THETA POWER (ADD ONLY THIS BLOCK)
# =========================
from scipy.signal import welch

SFREQ = 128

def compute_bandpower(segment, sf, band):
    low, high = band
    freqs, psd = welch(segment, sf, nperseg=256)
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[:, idx], axis=1)

print("\nComputing delta & theta power...")

delta_train = []
theta_train = []
delta_test = []
theta_test = []

# TRAIN
for seg in X_train:
    delta_train.append(compute_bandpower(seg, SFREQ, (0.5, 4)))
    theta_train.append(compute_bandpower(seg, SFREQ, (4, 8)))

# TEST
for seg in X_test:
    delta_test.append(compute_bandpower(seg, SFREQ, (0.5, 4)))
    theta_test.append(compute_bandpower(seg, SFREQ, (4, 8)))

# Convert
delta_train = np.array(delta_train, dtype=np.float32)
theta_train = np.array(theta_train, dtype=np.float32)
delta_test = np.array(delta_test, dtype=np.float32)
theta_test = np.array(theta_test, dtype=np.float32)

# SAVE
np.save("data/delta_train.npy", delta_train)
np.save("data/theta_train.npy", theta_train)
np.save("data/delta_test.npy", delta_test)
np.save("data/theta_test.npy", theta_test)

print("Delta & Theta saved successfully!")