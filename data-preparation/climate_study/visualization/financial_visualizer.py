"""
financial_visualizer.py

Basic visualization for financial data from Filing nodes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


class FinancialVisualizer:
    """
    Simple visualizations for financial data (revenue, assets, leverage, ROA).
    
    Example
    -------
    >>> from climate_study.visualization import FinancialVisualizer
    >>> fin_viz = FinancialVisualizer()
    >>> fin_viz.plot_overview(df_financial)
    """
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def plot_overview(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot 2x2 overview of financial metrics.
        
        - Top-left: Revenue distribution (log scale)
        - Top-right: Total assets distribution (log scale)
        - Bottom-left: Leverage distribution
        - Bottom-right: ROA distribution
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        df = df.copy()
        
        # Top-left: Revenue distribution
        ax = axes[0, 0]
        if 'revenue' in df.columns:
            valid = df['revenue'].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                ax.hist(np.log10(valid), bins=50, edgecolor='white', alpha=0.7)
                ax.set_xlabel('Log10(Revenue)')
                ax.set_ylabel('Frequency')
        ax.set_title('Revenue Distribution')
        
        # Top-right: Total assets distribution
        ax = axes[0, 1]
        if 'total_assets' in df.columns:
            valid = df['total_assets'].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                ax.hist(np.log10(valid), bins=50, edgecolor='white', alpha=0.7, color='green')
                ax.set_xlabel('Log10(Total Assets)')
                ax.set_ylabel('Frequency')
        ax.set_title('Total Assets Distribution')
        
        # Bottom-left: Leverage distribution
        ax = axes[1, 0]
        if 'leverage' in df.columns:
            valid = df['leverage'].dropna()
            valid = valid[(valid >= 0) & (valid <= 2)]  # Reasonable range
            if len(valid) > 0:
                ax.hist(valid, bins=50, edgecolor='white', alpha=0.7, color='orange')
                ax.axvline(valid.median(), color='red', linestyle='--', label=f'Median: {valid.median():.2f}')
                ax.set_xlabel('Leverage (Liabilities/Assets)')
                ax.set_ylabel('Frequency')
                ax.legend()
        ax.set_title('Leverage Distribution')
        
        # Bottom-right: ROA distribution
        ax = axes[1, 1]
        if 'roa' in df.columns:
            valid = df['roa'].dropna()
            valid = valid[(valid >= -0.5) & (valid <= 0.5)]  # Reasonable range
            if len(valid) > 0:
                ax.hist(valid, bins=50, edgecolor='white', alpha=0.7, color='purple')
                ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                ax.axvline(valid.median(), color='red', linestyle='--', label=f'Median: {valid.median():.2%}')
                ax.set_xlabel('ROA (Net Income/Assets)')
                ax.set_ylabel('Frequency')
                ax.legend()
        ax.set_title('ROA Distribution')
        
        plt.suptitle('Financial Data Overview', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_trends(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot median financial metrics over time.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        df = df.copy()
        df['year'] = df['year'].astype(int)
        
        # Aggregate by year
        yearly = df.groupby('year').agg({
            'revenue': 'median',
            'leverage': 'median',
            'roa': 'median'
        }).reset_index()
        
        # Revenue trend
        ax = axes[0]
        if 'revenue' in yearly.columns:
            ax.plot(yearly['year'], yearly['revenue'] / 1e6, marker='o')
            ax.set_xlabel('Year')
            ax.set_ylabel('Median Revenue ($M)')
            ax.set_title('Revenue Over Time')
        
        # Leverage trend
        ax = axes[1]
        if 'leverage' in yearly.columns:
            ax.plot(yearly['year'], yearly['leverage'], marker='o', color='orange')
            ax.set_xlabel('Year')
            ax.set_ylabel('Median Leverage')
            ax.set_title('Leverage Over Time')
        
        # ROA trend
        ax = axes[2]
        if 'roa' in yearly.columns:
            ax.plot(yearly['year'], yearly['roa'], marker='o', color='purple')
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Year')
            ax.set_ylabel('Median ROA')
            ax.set_title('ROA Over Time')
        
        plt.tight_layout()
        return fig
    
    def summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return summary statistics for financial variables.
        """
        cols = ['revenue', 'net_income', 'total_assets', 'total_liabilities', 'leverage', 'roa']
        cols = [c for c in cols if c in df.columns]
        
        return df[cols].describe().T
