# medical_classifier_training.py
"""
Advanced Medical Image Classification Model Training Script
Goal: Achieve 85%+ accuracy for medical image classification
Optimized for PyTorch with advanced techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import timm  # Excellent for pre-trained models

# Data Processing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
import cv2
from PIL import Image
import random
import os
from collections import Counter
import json
from tqdm import tqdm
import time

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable optimizations for RTX GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("CUDA not available")
print("All imports successful!")
#Configuration Class
class Config:
    # Data paths
    DATA_CSV = "urgent_care_images_master_final.csv"
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Advanced training parameters
    WARMUP_EPOCHS = 5
    T_MAX = 45
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    
    # Model architecture
    DROPOUT_RATE = 0.3
    MODEL_NAME = 'efficientnet_b3'
    
    # Training strategy
    USE_MIXED_PRECISION = True
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_WORKERS = 0
    
    # Ensemble
    N_FOLDS = 5
    
    # Output paths
    OUTPUT_DIR = "pytorch_model_outputs"
    MODEL_SAVE_PATH = "best_medical_classifier.pth"

# Advanced Model Architecture
class AdvancedMedicalClassifier(nn.Module):
    """Advanced medical image classifier with multiple architectures"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=8, dropout_rate=0.3, pretrained=True):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Advanced pooling and attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 8, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        avg_features = self.global_avg_pool(features).view(features.size(0), -1)
        max_features = self.global_max_pool(features).view(features.size(0), -1)
        
        # Apply attention to avg features
        attention_weights = self.attention(avg_features)
        attended_features = avg_features * attention_weights
        
        # Combine features
        combined_features = torch.cat([attended_features, max_features], dim=1)
        
        # Classify
        output = self.classifier(combined_features)
        
        return output

# Advanced Loss Functions
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing to prevent overconfidence"""
    
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        targets_smooth = torch.zeros_like(log_prob)
        targets_smooth.fill_(self.smoothing / (self.num_classes - 1))
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -targets_smooth * log_prob
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)

class CombinedLoss(nn.Module):
    """Combine multiple loss functions"""
    
    def __init__(self, num_classes, class_weights=None, focal_weight=0.6, ce_weight=0.4, smoothing=0.1):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2, weight=class_weights, reduction='mean')
        self.smooth_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing, reduction='mean')
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        return self.focal_weight * focal + self.ce_weight * smooth

# Dataset Class
class MedicalImageDataset(Dataset):
    """Advanced PyTorch dataset with augmentations"""
    
    def __init__(self, dataframe, class_to_idx, transforms=None):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.df.iloc[idx]['filepath']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = self.df.iloc[idx]['label']
        label_idx = self.class_to_idx[label]
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        else:
            # Basic preprocessing if no transforms
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, label_idx

# MixUp Implementation
class MixUpCriterion:
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# Transform Functions
def get_train_transforms(img_size=224):
    """Medical image safe augmentations"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.8
        ),
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1]),
        ], p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    """Minimal transforms for validation/test"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

# Training Functions
def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, scaler=None, mixup_criterion=None):
    """Train for one epoch with MixUp"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Apply MixUp with 60% probability
        if np.random.random() > 0.4 and mixup_criterion is not None:
            mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=0.2, device=device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(mixed_images)
                    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mixed_images)
                loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Calculate accuracy (use original targets for simplicity)
            acc1 = (outputs.argmax(dim=1) == targets).float().mean()
        else:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            acc1 = (outputs.argmax(dim=1) == targets).float().mean()
        
        optimizer.zero_grad()
        
        # Update metrics
        running_loss += loss.item()
        correct += acc1.item() * images.size(0)
        total += images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(i+1):.4f}',
            'Acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100 * correct / total

def validate(model, criterion, val_loader, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions for detailed analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return running_loss / len(val_loader), 100 * correct / total, all_preds, all_targets

def compute_pytorch_class_weights(labels, method='focal_inspired'):
    """Compute class weights for PyTorch"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    n_classes = len(class_counts)
    
    if method == 'focal_inspired':
        weights = []
        max_count = max(class_counts.values())
        for cls in sorted(class_counts.keys()):
            ratio = class_counts[cls] / max_count
            weight = (1 / ratio) ** 0.5
            weights.append(weight)
    
    return torch.FloatTensor(weights)

def save_model_for_app(model, class_names, class_to_idx, idx_to_class, test_accuracy, output_dir="app_models"):
    """Save model and metadata for app.py usage"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save complete model
    complete_model_path = os.path.join(output_dir, 'complete_model.pth')
    torch.save(model, complete_model_path)
    
    # Save model metadata
    metadata = {
        'model_name': 'efficientnet_b3',
        'num_classes': len(class_names),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'img_size': 224,
        'test_accuracy': test_accuracy,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'model_architecture': 'AdvancedMedicalClassifier',
        'dropout_rate': 0.3
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save transforms for inference
    transform_code = '''
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
'''
    
    transforms_path = os.path.join(output_dir, 'transforms.py')
    with open(transforms_path, 'w') as f:
        f.write(transform_code)
    
    print(f"Model saved for app usage at: {output_dir}")
    print(f"- Model: {model_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Transforms: {transforms_path}")
    
    return output_dir

def main():
    """Main training function"""
    print("Starting Advanced Medical Image Classification Training")
    print("="*60)
    
    # Setup
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and analyze data
    print("Loading data...")
    df = pd.read_csv(config.DATA_CSV)
    print(f"Dataset shape: {df.shape}")
    
    # Get unique classes and create label mapping
    all_classes = sorted(df['label'].unique())
    n_classes = len(all_classes)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    print(f"Classes ({n_classes}): {all_classes}")
    
    # Create stratified splits
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, stratify=train_df['label'], random_state=42
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Calculate class weights
    train_labels = train_df['label'].tolist()
    class_weights = compute_pytorch_class_weights(train_labels, method='focal_inspired')
    class_weights = class_weights.to(device)
    
    # Create transforms
    train_transforms = get_train_transforms(config.IMG_SIZE)
    val_transforms = get_val_transforms(config.IMG_SIZE)
    
    # Create datasets
    train_dataset = MedicalImageDataset(train_df, class_to_idx, transforms=train_transforms)
    val_dataset = MedicalImageDataset(val_df, class_to_idx, transforms=val_transforms)
    test_dataset = MedicalImageDataset(test_df, class_to_idx, transforms=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = AdvancedMedicalClassifier(
        model_name=config.MODEL_NAME,
        num_classes=n_classes,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = CombinedLoss(
        num_classes=n_classes,
        class_weights=class_weights,
        focal_weight=0.6,
        ce_weight=0.4,
        smoothing=config.LABEL_SMOOTHING
    )
    
    mixup_criterion = MixUpCriterion(criterion)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE * 0.3,  # Lower LR for fine-tuning
        weight_decay=config.WEIGHT_DECAY * 1.5,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-8
    )
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Training loop
    print("Starting training...")
    best_acc = 0.0
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch+1, scaler, mixup_criterion
        )
        
        # Validation
        val_loss, val_acc, val_preds, val_targets = validate(model, criterion, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'class_names': all_classes,
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
            }, os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_PATH))
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        
        # Early stopping
        if epoch > 20 and val_acc < best_acc - 5:
            print("Early stopping triggered")
            break
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_PATH))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final testing
    print("Testing final model...")
    test_loss, test_acc, test_preds, test_targets = validate(model, criterion, test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save model for app usage
    app_model_dir = save_model_for_app(
        model, all_classes, class_to_idx, idx_to_class, test_acc
    )
    
    # Save final results
    final_results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_acc,
        'model_name': config.MODEL_NAME,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'class_names': all_classes,
        'training_completed': True
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nTraining completed successfully!")
    print(f"Model ready for app.py at: {app_model_dir}")
    
    return test_acc, app_model_dir

if __name__ == "__main__":
    test_accuracy, model_directory = main()
    print(f"\nModel training completed with {test_accuracy:.2f}% test accuracy")
    print(f"App-ready files saved to: {model_directory}")