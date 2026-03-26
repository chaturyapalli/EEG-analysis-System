import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, accuracy_score

# =========================
# GRAPH FUNCTION
# =========================
def create_graph(sample):
    num_nodes = sample.shape[0]
    edge_index = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=sample, edge_index=edge_index)

# =========================
# MODEL
# =========================
class CNN_GAT(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(17, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        self.gat = GATConv(64, 16, heads=1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        outputs = []
        for sample in x:
            graph = create_graph(sample)
            out = self.gat(graph.x, graph.edge_index)
            out = torch.mean(out, dim=0)
            outputs.append(out)

        outputs = torch.stack(outputs)
        return self.fc(outputs)


# =========================
# 🚨 MAIN BLOCK (IMPORTANT FIX)
# =========================
if __name__ == "__main__":

    print("Loading data...")

    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # add noise only to train
    X_train = X_train + np.random.normal(0, 0.01, X_train.shape)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = CNN_GAT()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")

    epochs = 5
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("\nEvaluating...")

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            out = model(X_batch)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.numpy())

    print(classification_report(y_test.numpy(), np.array(preds)))

    accuracy = accuracy_score(y_test.numpy(), np.array(preds))
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), "models/cnn_gat_model.pth")

    print("\nDone")