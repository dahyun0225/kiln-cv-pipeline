print(f"{'>'*71}")
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import os
import numpy as np
from pathlib import Path

# data and parameter setup ---->
DATA_DIR = './train'
TEST_DIR = './test'
CSV_FILE = os.path.join(DATA_DIR, 'data.csv')
BATCH_SIZE = 30
EPOCHS = 21
LEARNING_RATE = 0.0001
IMG_SIZE = 384
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")
print(f"VERSION 43: EfficientNetV2-S | Batch={BATCH_SIZE}, Epochs={EPOCHS}")
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

full_train_dataset = KilnDataset(CSV_FILE, DATA_DIR, transform=train_transforms)

all_labels = df['label'].values
sample_weights = [class_weights[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
#<----


# model ---->
print(f"\n{'-'*60}")
print("Building EfficientNetV2-S model...")

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

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
)
#<----


# training ---->
print(f"\n{'-'*60}")
print("training...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0

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

    train_acc = 100 * train_correct / len(full_train_dataset)
    avg_loss = train_loss / len(train_loader)

    scheduler.step(avg_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'best_kiln_model_v43_full.pth')

print(f"\n{'='*60}")
print("Training complete")
#<----


# test predictions ---->
if Path(TEST_DIR).exists():
    print("Test data prediction...")

    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])
    print(f"{len(test_images)} test images")

    model.eval()
    predictions = []

    with torch.no_grad():
        for i, img_name in enumerate(test_images):
            img_path = os.path.join(TEST_DIR, img_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = val_transforms(image).unsqueeze(0).to(device)

            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_prob = probabilities[0, 1].item()

            index = img_name.replace('.png', '')

            predictions.append({
                'index': index,
                'score': predicted_prob
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_images)} images.")

    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv("submission43_full.csv", index=False)

    print(f"\nPredictions saved to submission43_full.csv")
else:
    print(f"\ndirectory not found: {TEST_DIR}")
#<----

print("EfficientNetV2 training done")
print(f"{'<'*72}")
