import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    print("⚠️ seaborn not found. Confusion matrix will be displayed without seaborn.")
    sns = None

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except ImportError:
    print("❗ scikit-learn is not installed. Please install it with `pip install scikit-learn`. Confusion matrix will not be shown.")
    confusion_matrix = ConfusionMatrixDisplay = None

# ------------ Perceiver Classifier ------------
class PerceiverClassifier(nn.Module):
    def __init__(self, input_dim=10, latent_dim=512, num_latents=128, num_classes=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, x):
        B, N, C = x.shape
        x_proj = self.input_proj(x)
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        latents, _ = self.cross_attn(latents, x_proj, x_proj)
        latents = latents.mean(dim=1)
        out = self.mlp(latents)
        return out


# ------------ Dataset ------------
class VoxelDataset(Dataset):
    def __init__(self, voxel_dirs, label_dict):
        self.voxel_files = []
        for path in voxel_dirs:
            self.voxel_files.extend(Path(path).glob("*.npy"))
        self.voxel_files = self.voxel_files[:50]  # Only use a small subset for quick training
        self.label_dict = label_dict

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        file = self.voxel_files[idx]
        voxel = np.load(file)
        if voxel.ndim == 5:
            voxel = voxel[0]
        voxel = voxel.reshape(-1, voxel.shape[-1])
        max_points = 50000
        if voxel.shape[0] > max_points:
            indices = np.random.choice(voxel.shape[0], max_points, replace=False)
            voxel = voxel[indices]
        voxel = torch.tensor(voxel, dtype=torch.float32)
        label = self.label_dict[file.stem]
        return voxel, torch.tensor(label, dtype=torch.long)


# ------------ Training Loop ------------
def train(model, train_loader, val_loader, device, epochs=1000, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_counts = [sum(label == i for _, label in train_loader.dataset) for i in range(8)]
    weights = torch.tensor([1 / (c + 1e-5) for c in class_counts], device=device)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += y.size(0)
                all_preds.extend(pred.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_losses.append(val_loss / len(val_loader))
        acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}, Val Acc = {acc:.2%}")

        with torch.no_grad():
            probs = torch.softmax(pred, dim=1)
            print("Sample prediction probs:", probs[0].cpu().numpy())

    if confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

    return train_losses, val_losses


# ------------ Plot Loss ------------
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("perceiver_loss.png")
    plt.show()


# ------------ Main  ------------
if __name__ == '__main__':
    print("👀 Preparing to load data...")
    voxel_root = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\0004")
    voxel_dirs = [str(p / "voxel_grids") for p in voxel_root.glob("user_0002_scene_*/") if (p / "voxel_grids").exists()]

    print(f"Searching in voxel_dirs: {voxel_dirs}")

    json_path = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\label_dict.json")

    with open(json_path, "r") as f:
        label_dict = json.load(f)

    dataset = VoxelDataset(voxel_dirs, label_dict)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerceiverClassifier().to(device)

    train_losses, val_losses = train(model, train_loader, val_loader, device)
    plot_losses(train_losses, val_losses)

    torch.save(model.state_dict(), "perceiver_model.pth")
    print("✅ Training complete and model saved.")
