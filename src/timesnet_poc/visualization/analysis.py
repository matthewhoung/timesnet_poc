"""
Visualization and analysis tools for TimesNet 2D transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple, List, Dict
import pandas as pd

from ..models.timesnet import TimesNet, FFTPeriodDiscovery


class TimesNetAnalyzer:
    """
    Analyzer for TimesNet model internals and 2D transformations.
    """
    
    def __init__(self, model: TimesNet, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.model.eval()
    
    def analyze_period_discovery(self, x: torch.Tensor, sample_idx: int = 0) -> Dict:
        """
        Analyze the FFT-based period discovery for a given sample.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            sample_idx: Which sample to analyze
            
        Returns:
            Dictionary with period analysis results
        """
        # Get period discovery module
        period_discovery = self.model.encoder_layers[0].period_discovery
        
        # Analyze periods
        with torch.no_grad():
            periods, amplitudes = period_discovery(x)
        
        # Convert to numpy for analysis
        periods_np = periods[sample_idx].cpu().numpy()
        amplitudes_np = amplitudes[sample_idx].cpu().numpy()
        
        # Get the original time series
        original_series = x[sample_idx, :, 0].cpu().numpy()
        
        # Perform full FFT analysis
        fft_result = np.fft.fft(original_series)
        frequencies = np.fft.fftfreq(len(original_series))
        amplitudes_full = np.abs(fft_result)
        
        results = {
            'original_series': original_series,
            'top_periods': periods_np,
            'top_amplitudes': amplitudes_np,
            'full_fft_frequencies': frequencies[:len(frequencies)//2],
            'full_fft_amplitudes': amplitudes_full[:len(amplitudes_full)//2],
            'dominant_period': periods_np[0] if len(periods_np) > 0 else None
        }
        
        return results
    
    def visualize_period_discovery(self, x: torch.Tensor, sample_idx: int = 0, save_path: str = None):
        """
        Create comprehensive visualization of period discovery process.
        """
        results = self.analyze_period_discovery(x, sample_idx)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Original Time Series',
                'FFT Frequency Spectrum',
                'Identified Periods',
                '2D Transformation Preview'
            ],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': True}]]
        )
        
        # Plot 1: Original time series
        fig.add_trace(
            go.Scatter(
                y=results['original_series'],
                mode='lines',
                name='Time Series',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Plot 2: FFT spectrum
        fig.add_trace(
            go.Scatter(
                x=results['full_fft_frequencies'],
                y=results['full_fft_amplitudes'],
                mode='lines',
                name='FFT Spectrum',
                line=dict(color='green', width=1)
            ),
            row=1, col=2
        )
        
        # Highlight top frequencies
        if len(results['top_periods']) > 0:
            top_frequencies = 1 / results['top_periods']
            for i, (freq, amp) in enumerate(zip(top_frequencies, results['top_amplitudes'])):
                if freq < max(results['full_fft_frequencies']):
                    fig.add_trace(
                        go.Scatter(
                            x=[freq],
                            y=[amp],
                            mode='markers',
                            name=f'Top {i+1}',
                            marker=dict(size=10, color='red')
                        ),
                        row=1, col=2
                    )
        
        # Plot 3: Identified periods
        if len(results['top_periods']) > 0:
            fig.add_trace(
                go.Bar(
                    x=[f'Period {i+1}' for i in range(len(results['top_periods']))],
                    y=results['top_periods'],
                    name='Period Length',
                    marker_color='purple'
                ),
                row=2, col=1
            )
        
        # Plot 4: 2D transformation preview (if dominant period exists)
        if results['dominant_period'] is not None:
            period = int(results['dominant_period'])
            if period > 1 and period < len(results['original_series']):
                # Reshape to 2D
                series = results['original_series']
                num_periods = len(series) // period
                if num_periods > 1:
                    reshaped_2d = series[:num_periods * period].reshape(num_periods, period)
                    
                    # Create heatmap data
                    fig.add_trace(
                        go.Heatmap(
                            z=reshaped_2d,
                            name='2D Transform',
                            colorscale='Viridis',
                            showscale=True
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title=f'TimesNet Period Discovery Analysis - Sample {sample_idx}',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Steps", row=1, col=1)
        fig.update_xaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Periods", row=2, col=1)
        fig.update_xaxes(title_text="Intraperiod Position", row=2, col=2)
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=2)
        fig.update_yaxes(title_text="Period Length", row=2, col=1)
        fig.update_yaxes(title_text="Interperiod (Time)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return results
    
    def visualize_2d_transformation(self, x: torch.Tensor, sample_idx: int = 0, save_path: str = None):
        """
        Visualize how 1D time series is transformed into 2D representations.
        """
        results = self.analyze_period_discovery(x, sample_idx)
        original_series = results['original_series']
        top_periods = results['top_periods']
        
        # Create figure with subplots for each period
        n_periods = min(len(top_periods), 4)  # Show top 4 periods
        fig, axes = plt.subplots(2, n_periods, figsize=(4*n_periods, 8))
        
        if n_periods == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_periods):
            period = int(top_periods[i])
            if period <= 1 or period >= len(original_series):
                continue
                
            # Calculate number of complete periods
            num_complete_periods = len(original_series) // period
            if num_complete_periods < 2:
                continue
            
            # Reshape to 2D
            truncated_length = num_complete_periods * period
            truncated_series = original_series[:truncated_length]
            reshaped_2d = truncated_series.reshape(num_complete_periods, period)
            
            # Plot 1D representation
            axes[0, i].plot(truncated_series, color='blue', linewidth=1)
            axes[0, i].set_title(f'1D Series (Period {period})')
            axes[0, i].set_xlabel('Time Steps')
            axes[0, i].set_ylabel('Value')
            axes[0, i].grid(True, alpha=0.3)
            
            # Add vertical lines to show period boundaries
            for p in range(1, num_complete_periods):
                axes[0, i].axvline(x=p*period, color='red', linestyle='--', alpha=0.5)
            
            # Plot 2D representation as heatmap
            im = axes[1, i].imshow(reshaped_2d, aspect='auto', cmap='viridis')
            axes[1, i].set_title(f'2D Transform ({num_complete_periods}×{period})')
            axes[1, i].set_xlabel('Intraperiod Position')
            axes[1, i].set_ylabel('Interperiod (Cycle Number)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_attention_patterns(self, x: torch.Tensor, layer_idx: int = 0) -> Dict:
        """
        Analyze attention patterns within TimesBlocks (conceptual, as our implementation 
        doesn't have explicit attention matrices like transformers).
        """
        # This is a placeholder for attention analysis
        # In a full implementation, you would extract attention weights
        # from the transformer components
        
        with torch.no_grad():
            # Get intermediate representations
            representations = []
            current_input = self.model.enc_embedding(x)
            
            for i, layer in enumerate(self.model.encoder_layers):
                current_input = layer(current_input)
                representations.append(current_input.clone())
                
                if i == layer_idx:
                    break
        
        return {
            'layer_representations': representations,
            'input_shape': x.shape,
            'output_shape': representations[-1].shape if representations else None
        }
    
    def compare_with_without_2d_transform(self, x: torch.Tensor, sample_idx: int = 0):
        """
        Compare the original series with its 2D transformation effects.
        """
        # Analyze the dominant period transformation
        results = self.analyze_period_discovery(x, sample_idx)
        original_series = results['original_series']
        dominant_period = results['dominant_period']
        
        if dominant_period is None or dominant_period <= 1:
            print("No valid dominant period found for comparison")
            return
        
        period = int(dominant_period)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original time series
        axes[0, 0].plot(original_series, color='blue', linewidth=2)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Period-aligned view
        num_complete_periods = len(original_series) // period
        if num_complete_periods > 1:
            truncated_length = num_complete_periods * period
            truncated_series = original_series[:truncated_length]
            
            for i in range(num_complete_periods):
                start_idx = i * period
                end_idx = (i + 1) * period
                period_data = truncated_series[start_idx:end_idx]
                axes[0, 1].plot(period_data, alpha=0.7, label=f'Period {i+1}')
            
            axes[0, 1].set_title(f'Overlaid Periods (Length {period})')
            axes[0, 1].set_xlabel('Intraperiod Position')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # 2D heatmap
            reshaped_2d = truncated_series.reshape(num_complete_periods, period)
            im1 = axes[1, 0].imshow(reshaped_2d, aspect='auto', cmap='viridis')
            axes[1, 0].set_title('2D Representation')
            axes[1, 0].set_xlabel('Intraperiod Position')
            axes[1, 0].set_ylabel('Interperiod (Cycle)')
            plt.colorbar(im1, ax=axes[1, 0])
            
            # Averaged period pattern
            mean_pattern = np.mean(reshaped_2d, axis=0)
            std_pattern = np.std(reshaped_2d, axis=0)
            
            axes[1, 1].fill_between(
                range(period), 
                mean_pattern - std_pattern, 
                mean_pattern + std_pattern, 
                alpha=0.3, color='blue', label='±1 STD'
            )
            axes[1, 1].plot(mean_pattern, color='red', linewidth=2, label='Mean Pattern')
            axes[1, 1].set_title('Average Intraperiod Pattern')
            axes[1, 1].set_xlabel('Intraperiod Position')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


def create_analysis_notebook():
    """
    Create a Jupyter notebook for interactive analysis.
    """
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TimesNet Analysis Notebook\\n",
    "Interactive analysis of TimesNet 2D transformations and period discovery."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import sys\\n",
    "import os\\n",
    "\\n",
    "# Add project root to path\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "from timesnet_poc.models.timesnet import TimesNet\\n",
    "from timesnet_poc.data.data_utils import create_financial_dataloader\\n",
    "from timesnet_poc.visualization.analysis import TimesNetAnalyzer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "train_loader, val_loader, test_loader, data_loader = create_financial_dataloader(\\n",
    "    symbols=['AAPL', 'MSFT', 'GOOGL'],\\n",
    "    target_symbol='AAPL',\\n",
    "    seq_len=96,\\n",
    "    pred_len=24,\\n",
    "    batch_size=16\\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create and load model\\n",
    "model = TimesNet(seq_len=96, pred_len=24, d_model=64, e_layers=2, top_k=5)\\n",
    "\\n",
    "# Load trained model if available\\n",
    "try:\\n",
    "    checkpoint = torch.load('../experiments/timesnet_financial/best_model.pt', map_location='cpu')\\n",
    "    model.load_state_dict(checkpoint['model_state_dict'])\\n",
    "    print('Loaded trained model')\\n",
    "except:\\n",
    "    print('Using untrained model for analysis')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get sample data\\n",
    "sample_batch = next(iter(test_loader))\\n",
    "sample_x, sample_y = sample_batch\\n",
    "print(f'Sample shape: {sample_x.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create analyzer\\n",
    "analyzer = TimesNetAnalyzer(model, data_loader)\\n",
    "\\n",
    "# Analyze period discovery\\n",
    "results = analyzer.visualize_period_discovery(sample_x, sample_idx=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize 2D transformations\\n",
    "analyzer.visualize_2d_transformation(sample_x, sample_idx=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare with and without 2D transformation\\n",
    "analyzer.compare_with_without_2d_transform(sample_x, sample_idx=0)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('notebooks/timesnet_analysis.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("Analysis notebook created at notebooks/timesnet_analysis.ipynb")


if __name__ == "__main__":
    create_analysis_notebook()