import mne
import numpy as np
import os
import matplotlib.pyplot as plt

TARGET_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4',
    'P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Cz'
]

MAX_DURATION = 300
PLOT_LIMIT = 10 


def clean_channel_name(ch):
    ch = ch.replace('EEG ', '').replace('-', '').strip()
    return ch.upper()


def preprocess_file(file_path, save_plot=False, plot_dir=None):

    print("\nProcessing:", file_path)

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Normalize channel names
    mapping = {ch: clean_channel_name(ch) for ch in raw.ch_names}
    raw.rename_channels(mapping)

    # BEFORE
    raw_before = raw.copy()
    raw_before.crop(tmin=0, tmax=10)
    data_before = raw_before.get_data()

    # PREPROCESSING
    raw.crop(tmin=0, tmax=min(MAX_DURATION, raw.times[-1]))

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing='ignore')

    raw.filter(0.5, 40)
    raw.notch_filter(50)

    raw.resample(128)
    raw.set_eeg_reference('average')

    # Channel selection
    target_upper = [ch.upper() for ch in TARGET_CHANNELS]
    available = [ch for ch in target_upper if ch in raw.ch_names]

    if len(available) >= 10:
        raw.pick_channels(available)
        print("Using standard 17 electrodes")
    else:
        print("Using all channels (fallback)")

    data_after = raw.get_data()

    # Normalize
    data_after = (data_after - np.mean(data_after)) / np.std(data_after)

    # Plot
    if save_plot and plot_dir is not None:

        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(data_before[0][:1000])
        plt.title("Before Preprocessing")

        plt.subplot(2, 1, 2)
        plt.plot(data_after[0][:1000])
        plt.title("After Preprocessing")

        plt.tight_layout()

        file_name = os.path.basename(file_path).replace(".edf", ".png")
        save_path = os.path.join(plot_dir, file_name)

        plt.savefig(save_path)
        plt.close()

    return data_after, data_before


def process_dataset(folder, dataset_name):

    dataset = []
    dataset_raw = []

    plot_count = 0
    plot_dir = os.path.join("plots", dataset_name)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".edf"):

                path = os.path.join(root, file)

                try:
                    save_plot = plot_count < PLOT_LIMIT

                    data_after, data_before = preprocess_file(
                        path,
                        save_plot=save_plot,
                        plot_dir=plot_dir
                    )

                    dataset.append(data_after)
                    dataset_raw.append(data_before)

                    if save_plot:
                        plot_count += 1

                except Exception as e:
                    print("Error:", file, e)

    return dataset, dataset_raw



# MAIN


print("\nProcessing OpenNeuro dataset")
openneuro, openneuro_raw = process_dataset("EEG/Open_Neuro", "OpenNeuro")

print("\nProcessing CHB-MIT dataset")
chbmit, chbmit_raw = process_dataset("EEG/CHB_MIT", "CHB_MIT")


# Save
os.makedirs("data", exist_ok=True)

np.save("data/X_openneuro.npy", np.array(openneuro, dtype=object), allow_pickle=True)
np.save("data/X_chbmit.npy", np.array(chbmit, dtype=object), allow_pickle=True)

np.save("data/X_openneuro_raw.npy", np.array(openneuro_raw, dtype=object), allow_pickle=True)
np.save("data/X_chbmit_raw.npy", np.array(chbmit_raw, dtype=object), allow_pickle=True)



# FINAL SUMMARY OUTPUT


print("\n Preprocessing Finished")

print("\n--- FINAL DATA SUMMARY ---")

print("\nOpenNeuro:")
print("Number of recordings:", len(openneuro))
if len(openneuro) > 0:
    print("Shape of one sample:", openneuro[0].shape)

print("\nCHB-MIT:")
print("Number of recordings:", len(chbmit))
if len(chbmit) > 0:
    print("Shape of one sample:", chbmit[0].shape)



# SAMPLE VALUES


def print_sample(name, dataset, raw_dataset):
    if len(dataset) > 0:
        print(f"\nSample Output ({name})")

        print("\nBEFORE (first 3 channels × first 5 samples):")
        print(raw_dataset[0][:3, :5])

        print("\nAFTER (first 3 channels × first 5 samples):")
        print(dataset[0][:3, :5])


print_sample("OpenNeuro", openneuro, openneuro_raw)
print_sample("CHB-MIT", chbmit, chbmit_raw)



# DATA SIZE


def dataset_size(dataset, name):
    if len(dataset) > 0:
        sample = dataset[0]
        total_bytes = sum(arr.nbytes for arr in dataset)
        size_mb = total_bytes / (1024 * 1024)

        print(f"\n{name} Dataset Size:")
        print("Sample shape:", sample.shape)
        print(f"Total size: {size_mb:.2f} MB")


dataset_size(openneuro, "OpenNeuro")
dataset_size(chbmit, "CHB-MIT")