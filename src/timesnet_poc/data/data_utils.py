"""
Financial data utilities for TimesNet proof of concept.
Handles data downloading, preprocessing, and dataset creation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class FinancialDataLoader:
    """
    Utility class for downloading and preprocessing financial data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        self.raw_data = None
        
    def download_data(
        self, 
        symbols: List[str] = None, 
        period: str = "2y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download financial data using yfinance.
        
        Args:
            symbols: List of stock symbols (default: major indices)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with financial data
        """
        if symbols is None:
            # Default to major indices and some volatile stocks for interesting patterns
            symbols = ["SPY", "QQQ", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]
        
        print(f"Downloading data for {symbols}...")
        
        data_dict = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                if not hist.empty:
                    data_dict[symbol] = hist
                    print(f"✓ Downloaded {symbol}: {len(hist)} records")
                else:
                    print(f"✗ No data for {symbol}")
            except Exception as e:
                print(f"✗ Error downloading {symbol}: {e}")
        
        # Combine data
        if data_dict:
            self.raw_data = data_dict
            # For simplicity, we'll focus on closing prices
            price_data = {}
            for symbol, data in data_dict.items():
                price_data[f"{symbol}_Close"] = data['Close']
                price_data[f"{symbol}_Volume"] = data['Volume']
                price_data[f"{symbol}_High"] = data['High']
                price_data[f"{symbol}_Low"] = data['Low']
            
            self.data = pd.DataFrame(price_data).dropna()
            print(f"Combined dataset shape: {self.data.shape}")
            return self.data
        else:
            raise ValueError("No data downloaded successfully")
    
    def create_features(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create technical indicators and features for time series analysis.
        """
        if data is None:
            data = self.data.copy()
        
        # Get close prices for main symbols
        close_cols = [col for col in data.columns if 'Close' in col]
        
        for col in close_cols:
            base_name = col.replace('_Close', '')
            
            # Returns
            data[f'{base_name}_Returns'] = data[col].pct_change()
            
            # Moving averages
            data[f'{base_name}_MA5'] = data[col].rolling(window=5).mean()
            data[f'{base_name}_MA20'] = data[col].rolling(window=20).mean()
            
            # Volatility (rolling standard deviation)
            data[f'{base_name}_Vol'] = data[col].rolling(window=20).std()
            
            # RSI (simplified)
            delta = data[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data[f'{base_name}_RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data[f'{base_name}_BB_upper'] = data[f'{base_name}_MA20'] + (data[f'{base_name}_Vol'] * 2)
            data[f'{base_name}_BB_lower'] = data[f'{base_name}_MA20'] - (data[f'{base_name}_Vol'] * 2)
        
        # Drop NaN values
        data = data.dropna()
        
        print(f"Features created. Dataset shape: {data.shape}")
        print(f"Feature columns: {data.columns.tolist()}")
        
        return data
    
    def prepare_time_series_data(
        self, 
        target_column: str,
        seq_len: int = 96,
        pred_len: int = 24,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for time series forecasting.
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
        
        # Select target and features
        if target_column not in self.data.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        # For this PoC, we'll use univariate prediction
        target_data = self.data[target_column].values.reshape(-1, 1)
        
        # Normalize the data
        self.scaler.fit(target_data)
        normalized_data = self.scaler.transform(target_data)
        
        # Create sequences
        X, y = self._create_sequences(normalized_data, seq_len, pred_len)
        
        # Split data
        total_samples = len(X)
        test_start = int(total_samples * (1 - test_size))
        val_start = int(test_start * (1 - val_size))
        
        X_train = X[:val_start]
        y_train = y[:val_start]
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
        seq_len: int, 
        pred_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for time series prediction.
        """
        X, y = [], []
        
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:(i + seq_len)])
            y.append(data[(i + seq_len):(i + seq_len + pred_len)])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized predictions back to original scale.
        """
        return self.scaler.inverse_transform(data)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_financial_dataloader(
    symbols: List[str] = None,
    target_symbol: str = "AAPL",
    seq_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 32,
    period: str = "2y"
) -> Tuple[DataLoader, DataLoader, DataLoader, FinancialDataLoader]:
    """
    Create DataLoaders for financial time series data.
    
    Returns:
        train_loader, val_loader, test_loader, data_loader_instance
    """
    # Initialize data loader
    data_loader = FinancialDataLoader()
    
    # Download and prepare data
    data_loader.download_data(symbols=symbols, period=period)
    featured_data = data_loader.create_features()
    
    # Prepare target column
    target_column = f"{target_symbol}_Close"
    
    # Prepare time series data
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.prepare_time_series_data(
        target_column=target_column,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, data_loader


# Example usage
if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, data_loader = create_financial_dataloader(
        symbols=["AAPL", "MSFT", "GOOGL"],
        target_symbol="AAPL",
        seq_len=96,
        pred_len=24,
        batch_size=16
    )
    
    print("DataLoaders created successfully!")
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        break