from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os
import mne
import sys


# LOAD MODEL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_model import CNN_GAT

model = CNN_GAT()
model.load_state_dict(torch.load("models/cnn_gat_model.pth", map_location="cpu"))
model.eval()


# ELECTRODES

electrodes = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
    'O1','O2','F7','F8','T3','T4','T5','T6','Cz'
]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    tmp = None
    try:
        file = request.files['file']

        if not file or not file.filename.endswith(".edf"):
            return jsonify({"error": "Upload valid .edf file"})

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        tmp.close()
        file.save(tmp.name)

        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
        raw.pick_types(eeg=True)

        sfreq = int(raw.info['sfreq'])
        data, _ = raw[:, :3000]
        data = data.astype(np.float32)

        
        # NORMALIZE EEG
        
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)

        # FIX CHANNEL SIZE
        if data.shape[0] > 17:
            data = data[:17, :]
        elif data.shape[0] < 17:
            pad = np.zeros((17 - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack((data, pad))

        
        # SEGMENT EEG (FAST VERSION)
        
        WINDOW_SIZE = sfreq * 5
        segments = []

        total_samples = data.shape[1]
        num_segments = total_samples // WINDOW_SIZE

        # Only process first 20 segments for speed
        max_segments = min(20, num_segments)

        for i in range(max_segments):
            seg = data[:, i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]
            if seg.shape[1] == WINDOW_SIZE:
                segments.append(seg)

        if len(segments) == 0:
            return jsonify({"error": "EEG too short"})

        segments = np.array(segments, dtype=np.float32)

        
        # MODEL PREDICTION
        
        abnormal_count = 0
        confs = []

        for seg in segments:
            seg_tensor = torch.tensor(seg).unsqueeze(0)

            with torch.no_grad():
                out = model(seg_tensor)
                prob = F.softmax(out, dim=1)
                conf, pred = torch.max(prob, dim=1)

                if pred.item() == 1:
                    abnormal_count += 1

                confs.append(conf.item())

        abnormal_ratio = abnormal_count / len(segments)
        confidence = float(np.mean(confs)) * 100

        
        # SIGNAL VARIANCE
        
        signal_variance = np.var(data)

        
        # DELTA / THETA POWER
        
        delta = float(np.mean(np.abs(segments[:, :, :100])))
        theta = float(np.mean(np.abs(segments[:, :, 100:200])))

        total_power = delta + theta + 1e-6
        delta_ratio = delta / total_power
        theta_ratio = theta / total_power

        def band_status(r):
            if r < 0.3:
                return "Low"
            elif r < 0.6:
                return "Medium"
            else:
                return "High"

        delta_status = band_status(delta_ratio)
        theta_status = band_status(theta_ratio)


        # WEIGHTED DECISION SYSTEM
        abnormal_score = 0
        sleep_score = 0
        normal_score = 0

        # --- Model output importance ---
        abnormal_score += abnormal_ratio * 3

        # --- Band analysis ---
        if delta_ratio > 0.65:
            sleep_score += 2

        if theta_ratio > 0.6:
            abnormal_score += 1

        if 0.35 <= delta_ratio <= 0.6:
            normal_score += 1

        if 0.35 <= theta_ratio <= 0.6:
            normal_score += 1

        # --- Signal variance ---
        if signal_variance > 5:
            abnormal_score += 1
        elif signal_variance < 4:
            normal_score += 1

        # --- Final decision ---
        if abnormal_score >= sleep_score and abnormal_score >= normal_score:
            prediction = "Abnormal"

        elif sleep_score >= abnormal_score and sleep_score >= normal_score:
            prediction = "Sleep EEG"

        elif normal_score >= abnormal_score and normal_score >= sleep_score:
            prediction = "Normal"

        else:
            prediction = "Borderline"

        # 🧠 ENCEPHALITIS DETECTION

        encephalitis_flag = False
        encephalitis_level = "None"

        if prediction == "Abnormal":

            if delta_ratio > 0.5 and theta_ratio > 0.4:
                encephalitis_flag = True
                encephalitis_level = "Strong"

            elif delta_ratio > 0.45 or theta_ratio > 0.4:
                encephalitis_flag = True
                encephalitis_level = "Moderate"

            else:
                encephalitis_level = "Mild"

        
        # IMPORTANT ELECTRODES
        
        sample = segments[0]
        activity = np.mean(np.abs(sample), axis=1)
        activity = activity / np.max(activity)

        top_idx = np.argsort(activity)[-3:]
        important_channels = [electrodes[i] for i in top_idx]
        channel_scores = activity.astype(float).tolist()

        signal = (sample[0][:200]).astype(float).tolist()
        
        # EEG INTERPRETATION TEXT

        interpretation = ""

        if encephalitis_flag:

            interpretation += f"Encephalitis-like pattern detected ({encephalitis_level}). "

            interpretation += (
                "Diffuse slowing is observed in EEG signals with elevated delta and theta activity. "
            )

            interpretation += f"Dominant activity observed in electrodes {', '.join(important_channels)}."

        elif prediction == "Abnormal":

            interpretation += (
                "Abnormal EEG activity detected, but no strong evidence of encephalitis-like patterns. "
            )

        elif prediction == "Sleep EEG":

            interpretation = "EEG pattern resembles physiological sleep activity with dominant slow waves."

        elif prediction == "Normal":

            interpretation = "EEG appears normal with no encephalitis-like patterns detected."

        else:

            interpretation = "Borderline EEG activity detected. Clinical correlation recommended."


        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "electrodes": important_channels,
            "delta": delta_status,
            "theta": theta_status,
            "delta_value": float(delta_ratio),
            "theta_value": float(theta_ratio),
            "signal": signal,
            "channel_names": electrodes,
            "channel_scores": channel_scores,
            "interpretation": interpretation,
            "encephalitis_detected": encephalitis_flag,
            "encephalitis_level": encephalitis_level
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        if tmp and os.path.exists(tmp.name):
            os.remove(tmp.name)

if __name__ == "__main__":
    app.run(debug=True)