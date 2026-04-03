# EEG Analysis System – Decision Support Tool

## Project Overview

This project is an EEG Analysis Decision Support System developed to analyze EEG EDF files and classify brain activity patterns into categories such as **Normal EEG, Sleep EEG, Abnormal EEG (Diffuse Slowing / Seizure Patterns), and Borderline EEG**.

The system combines **signal processing, band power analysis, and deep learning abnormal segment detection** to assist in EEG interpretation.

This project is intended as a **clinical decision support prototype**, not a medical diagnostic tool.

---
## Dataset Sources

The EEG datasets used in this project were obtained from publicly available EEG repositories.

### 1. CHB-MIT Scalp EEG Database

Source: https://physionet.org/content/chbmit/1.0.0/

The CHB-MIT dataset contains pediatric EEG recordings with seizure annotations and was used for epileptic EEG recordings and abnormal EEG pattern analysis.

### 2. OpenNeuro EEG Dataset (ds003555)

Source: https://openneuro.org/datasets/ds003555

The OpenNeuro ds003555 dataset contains EEG recordings stored in EDF format and organized using the BIDS (Brain Imaging Data Structure) format. These recordings were used as proxy normal EEG recordings and additional EEG data for model training and evaluation.

### Dataset Summary Used in This Project

| Dataset            | Purpose          | Recordings Used |
| ------------------ | ---------------- | --------------- |
| OpenNeuro ds003555 | Proxy Normal EEG | 29 recordings   |
| CHB-MIT            | Epileptic EEG    | 61 recordings   |

All datasets used in this project are publicly available for research and educational purposes.


## Workflow / System Flow

The system follows this pipeline:

1. Upload EEG EDF file
2. EEG Preprocessing (Filtering, Normalization)
3. Signal Segmentation
4. CNN-GAT Model detects abnormal EEG segments
5. Feature Extraction:

   * Delta Band Power
   * Theta Band Power
   * Signal Variance
   * Electrode Activity
6. Decision Logic Classification
7. Results displayed in Web UI with graphs

### Flow Diagram (Conceptual)

```
EDF File
   ↓
Preprocessing
   ↓
Segmentation
   ↓
CNN-GAT Model → Abnormal Segments
   ↓
Feature Extraction
   ↓
Decision Logic
   ↓
Final Classification
   ↓
Visualization (UI)
```

---

## EEG Classification Logic

The system classifies EEG based on the following conditions:

| Condition                         | Classification                                 |
| --------------------------------- | ---------------------------------------------- |
| Delta > 0.6 AND Theta > 0.6       | Abnormal (Diffuse Slowing / Encephalitis-like) |
| Abnormal Segment Ratio > 0.4      | Abnormal                                       |
| Signal Variance > 6               | Abnormal                                       |
| Delta > 0.65 and Theta low        | Sleep EEG                                      |
| Balanced Delta & Theta (0.35–0.6) | Normal EEG                                     |
| Otherwise                         | Borderline EEG                                 |

This logic simulates clinical EEG interpretation patterns.

---

## Datasets Used

We used publicly available EEG datasets:

### 1. CHB-MIT Scalp EEG Database (Seizure EEG)

https://physionet.org/content/chbmit/1.0.0/

### 2. OpenNeuro EEG Dataset

https://openneuro.org/

### 3. Sleep EDF Database

https://physionet.org/content/sleep-edfx/

These datasets were used to simulate:

* Normal EEG
* Sleep EEG
* Seizure EEG
* Diffuse slowing EEG patterns

---

## Project Folder Structure

```
FINAL_YEAR_PROJECT
│
├── backend
│   ├── app.py
│   ├── preprocessing
│   ├── models
│   └── results
│
├── templates
│   └── index.html
│
├── EEG (EDF datasets)
├── data (processed numpy files)
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### Step 1 – Create Virtual Environment

```
python -m venv venv
```

### Step 2 – Activate Environment

Windows:

```
venv\Scripts\activate
```

### Step 3 – Install Requirements

```
pip install -r requirements.txt
```

### Step 4 – Run the Flask Application

Go inside backend folder and run:

```
python app.py
```

### Step 5 – Open Web Interface

Open browser and go to:

```
http://127.0.0.1:5000
```

### Step 6 – Upload EDF File

1. Click **Choose File**
2. Select EEG `.edf` file
3. Click **Analyze**
4. View results and graphs

---

## Output Displayed in UI

The system displays:

* Prediction (Normal / Abnormal / Sleep / Borderline)
* Delta Band Relative Power
* Theta Band Relative Power
* Important Electrodes
* EEG Signal Graph
* Relative Power Graph
* Electrode Activity Graph

---

## Technologies Used

* Python
* Flask
* MNE (EEG Processing)
* PyTorch
* NumPy
* Chart.js
* HTML/CSS/JavaScript

---

## Project Type

This project is a **Machine Learning + Signal Processing based EEG Decision Support System**.

---


