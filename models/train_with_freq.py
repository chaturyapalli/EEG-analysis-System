import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import os

# =========================
# GRAPH FUNCTION
# =========================
def create_graph(sample):
    num_nodes = sample.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(max(0, i-2), min(num_nodes, i+3)):
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=sample, edge_index=edge_index)


# =========================
# IMPROVED MODEL (WITH FREQ LEARNING)
# =========================
class CNN_GAT_FREQ(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN (unchanged + stable)
        self.cnn = nn.Sequential(
            nn.Conv1d(17, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # SAME GAT
        self.gat = GATConv(64, 16, heads=1)

        # 🔥 NEW: Learnable frequency layer
        self.freq_layer = nn.Sequential(
            nn.Linear(34, 34),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # SAME final layer
        self.fc = nn.Linear(50, 2)

    def forward(self, x, delta, theta):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        outputs = []
        for sample in x:
            graph = create_graph(sample)
            out = self.gat(graph.x, graph.edge_index)
            out = torch.mean(out, dim=0)
            outputs.append(out)

        outputs = torch.stack(outputs)

        # 🔥 frequency processing
        freq = torch.cat((delta, theta), dim=1)
        freq = self.freq_layer(freq)

        # 🔥 slight weighting
        combined = torch.cat((outputs, 1.5 * freq), dim=1)

        return self.fc(combined)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Loading data...")

    X_train = np.load("data/X_train.npy").astype(np.float32)
    y_train = np.load("data/y_train.npy")

    X_test = np.load("data/X_test.npy").astype(np.float32)
    y_test = np.load("data/y_test.npy")

    delta_train = np.load("data/delta_train.npy").astype(np.float32)
    theta_train = np.load("data/theta_train.npy").astype(np.float32)

    delta_test = np.load("data/delta_test.npy").astype(np.float32)
    theta_test = np.load("data/theta_test.npy").astype(np.float32)

    # =========================
    # NORMALIZE FREQ FEATURES
    # =========================
    delta_mean, delta_std = delta_train.mean(), delta_train.std()
    theta_mean, theta_std = theta_train.mean(), theta_train.std()

    delta_train = (delta_train - delta_mean) / (delta_std + 1e-6)
    theta_train = (theta_train - theta_mean) / (theta_std + 1e-6)

    delta_test = (delta_test - delta_mean) / (delta_std + 1e-6)
    theta_test = (theta_test - theta_mean) / (theta_std + 1e-6)

    # =========================
    # AUGMENTATION
    # =========================
    X_train = X_train + np.random.normal(0, 0.02, X_train.shape)

    # =========================
    # TO TENSOR
    # =========================
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    delta_train = torch.tensor(delta_train, dtype=torch.float32)
    theta_train = torch.tensor(theta_train, dtype=torch.float32)

    delta_test = torch.tensor(delta_test, dtype=torch.float32)
    theta_test = torch.tensor(theta_test, dtype=torch.float32)

    # =========================
    # MODEL
    # =========================
    model = CNN_GAT_FREQ()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining improved frequency model...")

    epochs = 60
    batch_size = 32
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for i in range(0, len(X_train), batch_size):

            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            delta_batch = delta_train[i:i+batch_size]
            theta_batch = theta_train[i:i+batch_size]

            optimizer.zero_grad()

            outputs = model(X_batch, delta_batch, theta_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # =========================
    # LOSS CURVE
    # =========================
    plt.figure()
    plt.plot(losses)
    plt.title("Loss Curve (Freq Improved)")
    plt.savefig("loss_freq_improved.png")
    plt.close()

    print("\nEvaluating...")

    model.eval()
    preds = []
    probs_all = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):

            X_batch = X_test[i:i+batch_size]
            delta_batch = delta_test[i:i+batch_size]
            theta_batch = theta_test[i:i+batch_size]

            out = model(X_batch, delta_batch, theta_batch)

            probs = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)

            preds.extend(pred.numpy())
            probs_all.extend(probs.numpy())

    y_true = y_test.numpy()
    y_pred = np.array(preds)
    probs_all = np.array(probs_all)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    fpr, tpr, _ = roc_curve(y_true, probs_all)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC Score: {roc_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_gat_freq_improved.pth")

    print("\nDone")