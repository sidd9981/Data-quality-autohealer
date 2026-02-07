"""
Distribution Shift Detector using PyTorch
Detects when data distribution changes significantly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import sys
sys.path.append('.')


class QualityDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DistributionShiftDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.5):
        super(DistributionShiftDetector, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DistributionShiftDetectorTrainer:
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 learning_rate: float = 0.001, weight_decay: float = 0.01):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = DistributionShiftDetector(input_dim, hidden_dims, dropout=0.5).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print(f"Using device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 30, early_stopping_patience: int = 10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_model('models/detectors/distribution_shift_best.pth')
                print("  Saved best model")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.load_model('models/detectors/distribution_shift_best.pth')
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().cpu().numpy()
        
        if probs.ndim == 0:
            probs = np.array([probs])
        
        preds = (probs > 0.5).astype(int)
        return preds, probs
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    from src.data.bad_data_generator import BadDataGenerator
    from src.profilers.spark_profiler import SparkDataProfiler
    from src.profilers.feature_engineering import QualityFeatureEngineer
    import os
    
    print("Training Distribution Shift Detector...")
    
    os.makedirs('models/detectors', exist_ok=True)
    
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    
    X_list = []
    y_list = []
    
    # Generate distribution shift samples
    print("\nGenerating distribution shift samples...")
    for i in range(100):
        shift_factor = np.random.uniform(1.5, 3.0)
        clean_df, corrupted_df = generator.generate_quality_issue_dataset(
            n_rows=200,
            issue_type='distribution_shift',
            column=np.random.choice(['salary', 'age', 'experience_years']),
            shift_factor=shift_factor
        )
        profile = profiler.profile_dataset(corrupted_df, baseline_df=clean_df)
        X_list.append(profile)
        y_list.append(1)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/100 shift samples")
    
    # Generate clean samples
    print("Generating clean samples...")
    for i in range(100):
        clean_df = generator.generate_clean_dataset(n_rows=200)
        clean_df2 = generator.generate_clean_dataset(n_rows=200)
        profile = profiler.profile_dataset(clean_df, baseline_df=clean_df2)
        X_list.append(profile)
        y_list.append(0)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/100 clean samples")
    
    print(f"\nTotal samples: {len(X_list)}")
    
    engineer = QualityFeatureEngineer()
    X = engineer.fit_transform(X_list)
    y = np.array(y_list)
    
    print(f"Feature matrix shape: {X.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = QualityDataset(X_train, y_train)
    val_dataset = QualityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    trainer = DistributionShiftDetectorTrainer(input_dim=X.shape[1], hidden_dims=[64, 32])
    trainer.train(train_loader, val_loader, epochs=30, early_stopping_patience=10)
    
    print("\nFinal evaluation:")
    val_metrics = trainer.evaluate(val_loader)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    engineer.save('models/detectors/distribution_shift_feature_engineer.pkl')
    profiler.spark.stop()
    print("\nDistribution shift detector training complete!")