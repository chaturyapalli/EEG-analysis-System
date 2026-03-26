import mne
import numpy as np
import os

TARGET_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4',
    'P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Cz'
]

# use only first 5 minutes of EEG
MAX_DURATION = 300


def preprocess_file(file_path):

    print("\nProcessing:", file_path)

    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    # crop first 5 minutes
    raw.crop(tmin=0, tmax=min(MAX_DURATION, raw.times[-1]))

    raw.load_data()

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing='ignore')

    # filtering
    raw.filter(0.5, 40)
    raw.notch_filter(50)

    # resample
    raw.resample(128)

    raw.set_eeg_reference('average')

    channel_names = raw.ch_names

    available = [ch for ch in TARGET_CHANNELS if ch in channel_names]

    if len(available) >= 10:
        print("Using standard electrodes:", available)
        raw.pick_channels(available)
    else:
        print("Using bipolar channels:", channel_names)

    data = raw.get_data()

    data = data.astype(np.float32)

    # normalize
    data = (data - np.mean(data)) / np.std(data)

    return data


def process_dataset(folder):

    dataset = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(".edf"):

                path = os.path.join(root, file)

                try:
                    data = preprocess_file(path)
                    dataset.append(data)

                except Exception as e:
                    print("\nError processing:", file)
                    print(e)

    return dataset


print("\n==============================")
print("Processing OpenNeuro dataset")
print("==============================\n")

openneuro = process_dataset("EEG/Open_Neuro")


print("\n==============================")
print("Processing CHB-MIT dataset")
print("==============================\n")

chbmit = process_dataset("EEG/CHB_MIT")


# create data folder
os.makedirs("data", exist_ok=True)

np.save("data/X_openneuro.npy", np.array(openneuro, dtype=object), allow_pickle=True)
np.save("data/X_chbmit.npy", np.array(chbmit, dtype=object), allow_pickle=True)

print("\n===================================")
print("Preprocessing finished successfully")
print("===================================\n")

print("OpenNeuro recordings:", len(openneuro))
print("CHB-MIT recordings:", len(chbmit))