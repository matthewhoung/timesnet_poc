"""
Training script for TimesNet proof of concept.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
import json

from models.timesnet import TimesNet
from data.data_utils import create_financial_dataloader, FinancialDataLoader


class TimesNetTrainer:
    """
    Trainer class for TimesNet model.
    """
    
    def __init__(
        self,
        model: TimesNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        data_loader: FinancialDataLoader,
        device: str = None,
        learning_rate: float = 1e-3,
        save_dir: str = "experiments"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_loader = data_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, epochs: int = 50, patience: int = 10) -> Dict:
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        self.save_model("final_model.pt")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def test(self, model_path: str = None) -> Dict:
        """
        Test the model and return metrics.
        """
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        test_losses = []
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                test_losses.append(loss.item())
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        test_metrics = {
            'test_loss': np.mean(test_losses),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        print("Test Results:")
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.6f}")
        
        return test_metrics, predictions, targets
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray, num_samples: int = 5):
        """Plot prediction vs target for sample sequences."""
        # Convert back to original scale
        pred_original = self.data_loader.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        target_original = self.data_loader.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(predictions))):
            ax = axes[i]
            
            # Plot prediction and target
            ax.plot(target_original[i, :, 0], label='Target', linewidth=2, alpha=0.8)
            ax.plot(pred_original[i, :, 0], label='Prediction', linewidth=2, alpha=0.8)
            
            ax.set_title(f'Sample {i+1}: Prediction vs Target')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'predictions_plot.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training function."""
    # Configuration
    config = {
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'target_symbol': 'AAPL',
        'seq_len': 96,  # Look back 96 time steps (e.g., 96 days)
        'pred_len': 24,  # Predict next 24 time steps
        'batch_size': 32,
        'period': '2y',  # 2 years of data
        'epochs': 100,
        'learning_rate': 1e-3,
        'patience': 15,
        'd_model': 64,
        'e_layers': 2,
        'top_k': 5,
        'save_dir': 'experiments/timesnet_financial'
    }
    
    print("TimesNet Financial Forecasting - Proof of Concept")
    print("=" * 50)
    
    # Create data loaders
    print("Loading financial data...")
    train_loader, val_loader, test_loader, data_loader = create_financial_dataloader(
        symbols=config['symbols'],
        target_symbol=config['target_symbol'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        batch_size=config['batch_size'],
        period=config['period']
    )
    
    # Create model
    print("\nInitializing TimesNet model...")
    model = TimesNet(
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        d_model=config['d_model'],
        e_layers=config['e_layers'],
        top_k=config['top_k'],
        enc_in=1,  # Univariate
        c_out=1    # Univariate
    )
    
    # Create trainer
    trainer = TimesNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_loader=data_loader,
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir']
    )
    
    # Save configuration
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(epochs=config['epochs'], patience=config['patience'])
    
    # Plot training history
    trainer.plot_training_history()
    
    # Test model
    print("\nTesting model...")
    test_metrics, predictions, targets = trainer.test()
    
    # Plot predictions
    trainer.plot_predictions(predictions, targets, num_samples=5)
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {config['save_dir']}")


if __name__ == "__main__":
    main()