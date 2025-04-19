import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop, RandomRotation, RandomHorizontalFlip, ColorJitter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from PIL import Image
import numpy as np
import os
import pandas as pd
import json
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# actions
actions = {
    'reaching': 0,
    'grasping': 1,
    'lifting': 2,
    'holding': 3,
    'transporting': 4,
    'placing': 5,
    'releasing': 6,
    'nothing': 7,
}

# Step 1: Define Dataset
class ActionSequenceDataset(Dataset):
    def __init__(self, sequences, transform=None):
        """
        sequences: List of tuples [(images, labels), ...]
            - images: List of PIL images for a sequence
            - labels: List of integer labels for each frame
        transform: Image transformations for preprocessing
        """
        self.sequences = sequences
        self.transform = transform
        self.img_root = r"D:\Github\ICRA_25_workshop\dataset_preparation\dataset\dataset_depth_Copy_full\human\cam_104122061850\pick"
        # Load the CSV file with labels
        self.df_labels = pd.read_csv(r"D:\Github\ICRA_25_workshop\dataset_preparation\dataset\dataset_depth_Copy_full\human\cam_104122061850\pick\data_augmentes_new.csv")
        self.allowed_ids = set(self.df_labels["Timestamp"].astype(str))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user_scene_img = self.sequences[idx]
     
        images_all = []
        labels_all = []
        if self.transform:
            for spec in user_scene_img:
                timestamp = spec[2].split(".")[0]
                if timestamp not in self.allowed_ids:
                    continue
                label = self.df_labels[self.df_labels['Timestamp'] == str(timestamp)]['ID'].values
                img_path = os.path.join(self.img_root, spec[0], spec[1], 'rgb', spec[2])
                img = Image.open(img_path).convert("RGB")
                image = self.transform(img) 
                
                images_all.append(image)
                labels_all.append(actions[label[0]])
        images = torch.stack(images_all)  # Shape: seq_len x 3 x 224 x 224
        labels = torch.tensor(labels_all, dtype=torch.long)

        return images, labels
    

# Step 2: Custom collate function for feature extraction and dataset loading
def collate_fn_extract(batch):
    # batch: List of (images, labels) where images is a list of tensors
    images = [item[0] for item in batch]  # List of lists of image tensors
    labels = [item[1] for item in batch]  # List of label tensors
    return images, labels

def collate_fn(batch):
    features = [item[0] for item in batch]  # List of feature tensors
    labels = [item[1] for item in batch]   # List of label tensors
    lengths = torch.tensor([len(f) for f in features])
    padded_features = pad_sequence(features, batch_first=True)  # batch_size x max_len x 512
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # -1 for ignored labels
    return padded_features, padded_labels, lengths


# Step 3: Feature Extraction with ResNet18
def extract_features(sequences, batch_size=1):
    # Load pretrained ResNet18
    resnet = models.resnet18(pretrained=True).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    resnet.eval()

    # # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = ActionSequenceDataset(sequences, transform=transform)
    print(f"Number of sequences: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_extract)

    features_list = []
    labels_list = []

    print(f"Number of batches: {len(dataloader)}")

    with torch.no_grad():

        for batch_images, batch_labels in tqdm(dataloader, desc="Extracting Features", unit="batch"):

             # Process each sequence in the batch
            batch_features = []
            for images in batch_images:
                # images is already a tensor of shape [seq_len, 3, 224, 224]
                images = images.to(device)
                # Process each image in the sequence
                seq_features = []
                for img in images:
                    # Add batch dimension for single image
                    img = img.unsqueeze(0)  # Shape: 1 x 3 x 224 x 224
                    # Extract features
                    feat = resnet(img)  # Shape: 1 x 512 x 1 x 1
                    feat = feat.view(512)  # Flatten to 512
                    seq_features.append(feat)
                # Stack sequence features
                seq_features = torch.stack(seq_features)  # Shape: seq_len x 512
                batch_features.append(seq_features)
            
            # Extend lists with processed features and labels
            features_list.extend(batch_features)
            labels_list.extend(batch_labels)

    return list(zip(features_list, labels_list))  # List of (features, labels) per sequence


# Step 4: Define Model (LSTM + MLP)
class ActionClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=8, dropout=0.7):
        super(ActionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x, lengths):
        # x: batch_size x seq_len x input_size
        # lengths: tensor of sequence lengths
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)  # batch_size x max_len x hidden_size

        preds = self.mlp(self.dropout(unpacked))
        return preds
    

# Step 5: Training Function
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0001, patience=200):
    class_counts = [1580, 680, 572, 448, 704, 544, 256, 4836]
    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)  # Ignore padded labels
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded labels
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_loss = float('inf')
    patience_counter = 0

    train_loss_all = []
    val_loss_all = []
    # Use tqdm for progress tracking during training
    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        model.train()
        train_loss = 0
        for features, labels, lengths in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(features, lengths)  # batch_size x max_len x 8
   
            loss = criterion(preds.view(-1, 8), labels.view(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_all.append(train_loss / len(train_loader))


        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features, labels = features.to(device), labels.to(device)
                # print(features.shape, labels.shape, lengths.shape)
                preds = model(features, lengths)
                loss = criterion(preds.view(-1, 8), labels.view(-1))
                val_loss += loss.item()
                _, predicted = torch.max(preds, dim=2)  # batch_size x max_len
                mask = labels != -1  # Ignore padded labels
                correct += (predicted[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        val_loss_all.append(val_loss / len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_loss_all, val_loss_all, model


# Step 6: Function to get sequence labels
def get_sequence_label(data_root):
    ids = ['0004', '0035', '0038']
    user_scene_img_all = []

    for id in (os.listdir(data_root)):
        if id not in ids:
            continue
        id_path = os.path.join(data_root, id)
        for file in (os.listdir(id_path)):
            user_scene_img = []
            if not file.endswith(".csv"):
                file_path = os.path.join(id_path, file)
                if not os.path.isdir(file_path):  # Ensure it's a directory before listing contents
                    print(f"Skipping non-directory: {file_path}")
                    continue

                for type in os.listdir(file_path):
                    if type == "rgb":
                        type_path = os.path.join(file_path, type)
                        img_dirs = os.listdir(type_path)
                        # Calculate halfway point
                        total_imgs = len(img_dirs)
                        halfway_point = total_imgs // 2  # Integer division for halfway point
                        for k, img in enumerate(img_dirs):
                            image_path = os.path.join(type_path, img)
                            
                            user_scene = image_path.split("\\")[-3]
                            img_name = image_path.split("\\")[-1]
                            user_scene_img.append((id, user_scene, img_name))
    
                user_scene_img_all.append(user_scene_img)
    return user_scene_img_all



# Step 7: Main Execution
def main():
    # Example: Load your data (replace with your actual data loading logic)
    # sequences = [(images_seq1, labels_seq1), (images_seq2, labels_seq2), ...]
    # images_seq: List of PIL Images, labels_seq: List of integers (0-6)
    data_root = r"D:\Github\ICRA_25_workshop\dataset_preparation\dataset\dataset_depth_copy_full\human\cam_104122061850\pick"
    
    pt_file = os.path.join(data_root, "feature_sequences.pt")

    feature_rebuild = False

    if os.path.exists(pt_file) and not feature_rebuild:
        print("Loading feature sequences from saved file...")
        loaded_data = torch.load(pt_file)
        feature_sequences = [(features.to(device), labels.to(device)) for features, labels in loaded_data]
    else:
        print("Extracting features with ResNet18...")
        sequences = get_sequence_label(data_root)
        feature_sequences = extract_features(sequences)
        data_cpu = [(features.cpu(), labels.cpu()) for features, labels in feature_sequences]

        torch.save(data_cpu, pt_file)
        print(f"Saved feature sequences to {pt_file}")


    random.shuffle(feature_sequences)
    train_val_size = int(0.9 * len(feature_sequences))
    train_val_seqs = feature_sequences[:train_val_size]
    test_seqs = feature_sequences[train_val_size:]

    # K-Fold Cross-Validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    batchsize = 16
    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_seqs)):
        print(f"Fold {fold+1}/{k_folds}")
        train_seqs = [train_val_seqs[i] for i in train_idx]
        val_seqs = [train_val_seqs[i] for i in val_idx]

        train_loader = DataLoader(train_seqs, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_seqs, batch_size=batchsize, shuffle=False, collate_fn=collate_fn)

        model = ActionClassifier(input_size=512, hidden_size=128, num_classes=8).to(device)
        train_loss, val_loss, model_trained = train_model(model, train_loader, val_loader, num_epochs=500, patience=50)
        
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
    
    # Debug: Print the lengths of the loss arrays to confirm variability
    print("Lengths of train losses:", [len(loss) for loss in all_train_losses])
    print("Lengths of val losses:", [len(loss) for loss in all_val_losses])

    # Pad the loss arrays to the same length
    max_len = max(len(loss) for loss in all_train_losses)  # Find the longest array
    padded_train_losses = []
    padded_val_losses = []

    for train_loss, val_loss in zip(all_train_losses, all_val_losses):
        # Pad training loss
        train_loss_array = np.array(train_loss)
        if len(train_loss_array) < max_len:
            pad_size = max_len - len(train_loss_array)
            padded_train_loss = np.pad(train_loss_array, (0, pad_size), mode='edge')  # Pad with the last value
        else:
            padded_train_loss = train_loss_array
        padded_train_losses.append(padded_train_loss)

        # Pad validation loss
        val_loss_array = np.array(val_loss)
        if len(val_loss_array) < max_len:
            pad_size = max_len - len(val_loss_array)
            padded_val_loss = np.pad(val_loss_array, (0, pad_size), mode='edge')  # Pad with the last value
        else:
            padded_val_loss = val_loss_array
        padded_val_losses.append(padded_val_loss)

    # Convert to numpy arrays and compute the mean
    padded_train_losses = np.array(padded_train_losses)
    padded_val_losses = np.array(padded_val_losses)
    avg_train_loss = np.mean(padded_train_losses, axis=0)
    avg_val_loss = np.mean(padded_val_losses, axis=0)

    # Plot average training and validation loss across folds
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(avg_train_loss) + 1), avg_train_loss, label='Training Loss')
    plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Training and Validation Loss (Cross-Validation)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_cv.png')
    plt.show()

    correct = 0
    total = 0
    # Inference example (on test set)
    model_trained.eval()
    with torch.no_grad():
        for features, labels in test_seqs:
            features = features.unsqueeze(0).to(device)  # Add batch dimension
            lengths = torch.tensor([len(features[0])])
            preds = model_trained(features, lengths)
            _, predicted = torch.max(preds, dim=2)

            labels = labels.squeeze(0).cpu().numpy()
            predicted = predicted.squeeze(0).cpu().numpy()
            print("Predicted labels:", predicted)
            print("True labels:", labels)

            correct += (predicted == labels).sum().item()
            total += len(labels)

        print(f'Accuracy: {(correct*100) / total:.2f} %')

if __name__ == "__main__":
    main()