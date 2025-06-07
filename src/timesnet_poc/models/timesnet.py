"""
TimesNet: Temporal 2D-Variation Modeling Implementation
A proof of concept implementation based on the ICLR 2023 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math


class FFTPeriodDiscovery(nn.Module):
    """
    Fast Fourier Transform-based period discovery module.
    Identifies the top-k most significant periods in the time series.
    """
    
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input time series [batch_size, seq_len, features]
        
        Returns:
            periods: Top-k periods [batch_size, top_k]
            amplitudes: Corresponding amplitudes [batch_size, top_k]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply FFT to the first feature (can be extended to multivariate)
        fft_result = torch.fft.fft(x[:, :, 0])  # [batch_size, seq_len]
        amplitudes = torch.abs(fft_result)[:, 1:seq_len//2]  # Remove DC and negative frequencies
        
        # Find top-k frequencies
        top_k_actual = min(self.top_k, amplitudes.shape[1])
        top_amplitudes, top_indices = torch.topk(amplitudes, top_k_actual, dim=1)
        
        # Convert frequency indices to periods
        periods = seq_len / (top_indices + 1).float()  # +1 because we removed DC component
        
        return periods, top_amplitudes


class TimesBlock(nn.Module):
    """
    Core TimesBlock that performs 2D transformation and processing.
    """
    
    def __init__(
        self, 
        seq_len: int, 
        d_model: int, 
        d_ff: int = None,
        top_k: int = 5,
        num_kernels: int = 6
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.top_k = top_k
        self.num_kernels = num_kernels
        
        # Period discovery
        self.period_discovery = FFTPeriodDiscovery(top_k)
        
        # Inception-like conv block for 2D processing
        self.inception_block = InceptionBlock(d_model, num_kernels)
        
        # Adaptive aggregation
        self.aggregation = nn.Linear(top_k * d_model, d_model)
        
        # Layer norm and residual
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            output: Processed tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Discover periods
        periods, amplitudes = self.period_discovery(x)
        
        # Process each period
        period_outputs = []
        for i in range(self.top_k):
            if i < periods.shape[1]:
                period_length = periods[:, i].int()
                # Handle variable period lengths by using the mean period
                avg_period = int(period_length.float().mean().item())
                avg_period = max(1, min(avg_period, seq_len))
                
                # Reshape to 2D tensor
                x_2d = self.reshape_to_2d(x, avg_period)
                
                # Apply inception block
                processed_2d = self.inception_block(x_2d)
                
                # Reshape back to 1D
                processed_1d = self.reshape_to_1d(processed_2d, seq_len)
                period_outputs.append(processed_1d)
            else:
                # Pad with zeros if we have fewer periods than top_k
                period_outputs.append(torch.zeros_like(x))
        
        # Aggregate period representations
        aggregated = torch.cat(period_outputs, dim=-1)  # [batch_size, seq_len, top_k * d_model]
        output = self.aggregation(aggregated)  # [batch_size, seq_len, d_model]
        
        # Residual connection and normalization
        output = self.norm(x + output)
        
        return output
    
    def reshape_to_2d(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """
        Reshape 1D time series to 2D tensor based on period.
        """
        batch_size, seq_len, d_model = x.shape
        
        if period <= 0 or period > seq_len:
            period = seq_len
        
        # Calculate number of complete periods
        num_periods = seq_len // period
        if num_periods == 0:
            num_periods = 1
            period = seq_len
        
        # Truncate to fit complete periods
        truncated_len = num_periods * period
        x_truncated = x[:, :truncated_len, :]
        
        # Reshape to 2D: [batch_size, num_periods, period, d_model]
        x_2d = x_truncated.view(batch_size, num_periods, period, d_model)
        
        # Rearrange for conv2d: [batch_size * d_model, 1, num_periods, period]
        x_2d = x_2d.permute(0, 3, 1, 2).contiguous()
        x_2d = x_2d.view(batch_size * d_model, 1, num_periods, period)
        
        return x_2d
    
    def reshape_to_1d(self, x_2d: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Reshape 2D tensor back to 1D time series.
        """
        batch_size_d_model, _, num_periods, period = x_2d.shape
        d_model = self.d_model
        batch_size = batch_size_d_model // d_model
        
        # Reshape back: [batch_size, d_model, num_periods, period]
        x_2d = x_2d.view(batch_size, d_model, num_periods, period)
        
        # Rearrange: [batch_size, num_periods, period, d_model]
        x_2d = x_2d.permute(0, 2, 3, 1).contiguous()
        
        # Flatten temporal dimensions: [batch_size, num_periods * period, d_model]
        seq_len_actual = num_periods * period
        x_1d = x_2d.view(batch_size, seq_len_actual, d_model)
        
        # Handle length mismatch
        if seq_len_actual < target_len:
            # Pad with zeros
            padding = torch.zeros(batch_size, target_len - seq_len_actual, d_model, 
                                device=x_1d.device, dtype=x_1d.dtype)
            x_1d = torch.cat([x_1d, padding], dim=1)
        elif seq_len_actual > target_len:
            # Truncate
            x_1d = x_1d[:, :target_len, :]
        
        return x_1d


class InceptionBlock(nn.Module):
    """
    Inception-style block for multi-scale 2D convolution processing.
    """
    
    def __init__(self, d_model: int, num_kernels: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_kernels = num_kernels
        
        # Different kernel sizes for multi-scale processing
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0)),
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(1, 2)),
        ])
        
        # Aggregation
        self.aggregation = nn.Conv2d(num_kernels, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 2D tensor [batch_size * d_model, 1, height, width]
        """
        # Apply different conv kernels
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = F.relu(conv_layer(x))
            conv_outputs.append(conv_out)
        
        # Concatenate along channel dimension
        combined = torch.cat(conv_outputs, dim=1)  # [batch_size * d_model, num_kernels, height, width]
        
        # Aggregate
        output = self.aggregation(combined)  # [batch_size * d_model, 1, height, width]
        
        return output


class TimesNet(nn.Module):
    """
    Complete TimesNet model for time series analysis.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        d_ff: int = None,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        enc_in: int = 1,
        c_out: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.enc_in = enc_in
        self.c_out = c_out
        
        # Input embedding
        self.enc_embedding = nn.Linear(enc_in, d_model)
        
        # TimesBlocks
        self.encoder_layers = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(e_layers)
        ])
        
        # Prediction head
        self.projection = nn.Linear(d_model, c_out)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series [batch_size, seq_len, enc_in]
        
        Returns:
            predictions: [batch_size, pred_len, c_out]
        """
        # Input embedding
        enc_out = self.enc_embedding(x)  # [batch_size, seq_len, d_model]
        enc_out = self.dropout(enc_out)
        
        # Process through TimesBlocks
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # Take the last pred_len steps for forecasting
        if self.pred_len <= self.seq_len:
            output = enc_out[:, -self.pred_len:, :]
        else:
            # If prediction length is longer than sequence length, repeat the last value
            last_values = enc_out[:, -1:, :].repeat(1, self.pred_len - self.seq_len, 1)
            output = torch.cat([enc_out[:, -(self.seq_len):, :], last_values], dim=1)
        
        # Project to output dimension
        output = self.projection(output)  # [batch_size, pred_len, c_out]
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    batch_size = 8
    seq_len = 96
    pred_len = 24
    enc_in = 1
    c_out = 1
    
    # Create model
    model = TimesNet(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=64,
        e_layers=2,
        top_k=5,
        enc_in=enc_in,
        c_out=c_out
    )
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, enc_in)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("TimesNet model created successfully!")