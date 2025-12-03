print(f"{'>'*71}")
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image
import pandas as pd
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# data and parameter setup ---->
DATA_DIR = './train'
TEST_DIR = './test'
CSV_FILE = os.path.join(DATA_DIR, 'data.csv')
BATCH_SIZE = 30
EPOCHS = 35
LEARNING_RATE = 0.0001
IMG_SIZE = 384
SEED = 42
N_FOLDS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")
print(f"VERSION 411_5cv: 5-Fold CV EfficientNetV2-S | Batch={BATCH_SIZE}, Epochs={EPOCHS}")
#<----


class KilnDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['index'] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# augmentation ---->
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#<----


# data ---->
df = pd.read_csv(CSV_FILE)
total_size = len(df)

class_counts = df['label'].value_counts().sort_index()
print(class_counts)
for cls in [0, 1]:
    print(f"  Class {cls}: {class_counts[cls]/len(df)*100:.1f}%")

class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(len(class_counts))])
class_weights = class_weights / class_weights.sum() * len(class_weights)
print(f"\nClass weights: {class_weights.numpy()}")
#<----


def build_model():
    model = models.efficientnet_v2_s(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    return model.to(device)


def train_fold(fold_idx, train_indices, val_indices):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_dataset_full = KilnDataset(CSV_FILE, DATA_DIR, transform=train_transforms)
    val_dataset_full = KilnDataset(CSV_FILE, DATA_DIR, transform=val_transforms)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_labels = [df.iloc[i]['label'] for i in train_indices]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_probs = []
        train_labels_list = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            train_probs.extend(probs)
            train_labels_list.extend(labels.cpu().numpy())

        train_acc = 100 * train_correct / len(train_dataset)
        train_auc = roc_auc_score(train_labels_list, train_probs)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / len(val_dataset)
        val_auc = roc_auc_score(val_labels_list, val_probs)

        scheduler.step(val_auc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, f'best_kiln_model_5cv_fold{fold_idx+1}.pth')
            print(f"  new best (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFold {fold_idx+1} complete - Best Val AUC: {best_val_auc:.4f}")
    return best_val_auc


# 5-fold cross-validation ---->
print(f"\n{'='*60}")
print("5-Fold Cross-Validation Training...")

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_aucs = []

for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(total_size))):
    fold_auc = train_fold(fold_idx, train_indices, val_indices)
    fold_aucs.append(fold_auc)

print(f"\n{'='*60}")
print("All folds trained!")
for i, auc in enumerate(fold_aucs):
    print(f"  Fold {i+1}: {auc:.4f}")
print(f"  Mean CV AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
#<----


# test predictions from CV models ---->
if Path(TEST_DIR).exists():
    print(f"\n{'='*60}")
    print("test predictions from CV models...")

    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])
    print(f"{len(test_images)} test images")

    fold_predictions = []

    for fold_idx in range(N_FOLDS):
        print(f"\n  Processing Fold {fold_idx+1}...")

        model = build_model()
        checkpoint = torch.load(f'best_kiln_model_5cv_fold{fold_idx+1}.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        predictions = []

        with torch.no_grad():
            for img_name in test_images:
                img_path = os.path.join(TEST_DIR, img_name)
                image = Image.open(img_path).convert('RGB')
                image_tensor = val_transforms(image).unsqueeze(0).to(device)

                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_prob = probabilities[0, 1].item()
                predictions.append(predicted_prob)

        fold_predictions.append(predictions)
        print(f"    Fold {fold_idx+1} predictions complete")

    cv_ensemble_scores = np.mean(fold_predictions, axis=0)

    print(f"\n  Loading full-data model from version 43...")
    full_model = build_model()
    full_checkpoint = torch.load('best_kiln_model_v43_full.pth', weights_only=False)
    full_model.load_state_dict(full_checkpoint['model_state_dict'])
    full_model.eval()

    full_predictions = []
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(TEST_DIR, img_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = val_transforms(image).unsqueeze(0).to(device)

            output = full_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_prob = probabilities[0, 1].item()
            full_predictions.append(predicted_prob)

    print(f"    Full model predictions complete")

    all_predictions = fold_predictions + [full_predictions]
    full_ensemble_scores = np.mean(all_predictions, axis=0)

    test_indices = [Path(img_name).stem for img_name in test_images]

    df_cv = pd.DataFrame({
        'index': test_indices,
        'score': cv_ensemble_scores
    })
    df_cv.to_csv("submission411_5cv_s.csv", index=False)

    print(f"\n{'='*60}")
    print(f"CV-only ensemble saved: submission411_5cv_s.csv")
    print(f"  Images: {len(df_cv)}")
    print(f"  Score stats - Mean: {cv_ensemble_scores.mean():.4f}, Std: {cv_ensemble_scores.std():.4f}")
    print(f"  Min: {cv_ensemble_scores.min():.4f}, Max: {cv_ensemble_scores.max():.4f}")

    df_full = pd.DataFrame({
        'index': test_indices,
        'score': full_ensemble_scores
    })
    df_full.to_csv("submission411_5cv_f.csv", index=False)

    print(f"\nFull ensemble saved: submission411_5cv_f.csv")
    print(f"  Images: {len(df_full)}")
    print(f"  Score stats - Mean: {full_ensemble_scores.mean():.4f}, Std: {full_ensemble_scores.std():.4f}")
    print(f"  Min: {full_ensemble_scores.min():.4f}, Max: {full_ensemble_scores.max():.4f}")

    print(f"\n{'='*60}")
    print("Sample predictions:")
    print(df_cv.head(10))
else:
    print(f"\ndirectory not found: {TEST_DIR}")
#<----

print(f"\n{'='*60}")
print("VERSION 411_5cv complete")
print(f"{'<'*72}")
