"""
location_visualizer.py

Visualization tools for firm property/location data.

Provides plots for understanding:
- Property type distribution
- Geographic distribution of properties
- Ownership breakdown
- Property size analysis
- Headquarters locations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


class LocationVisualizer:
    """
    Visualize firm property and location data.
    
    Provides diagnostic plots for understanding:
    - Property type distribution
    - Geographic distribution (states, cities)
    - Ownership breakdown
    - Property size analysis
    - Headquarters locations
    
    Example
    -------
    >>> from climate_study.visualization import LocationVisualizer
    >>> viz = LocationVisualizer()
    >>> viz.plot_overview(df_location)
    """
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def plot_overview(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Plot 2x2 overview of property location data.
        
        Includes:
        - Top-left: Property type distribution
        - Top-right: Ownership breakdown pie chart
        - Bottom-left: Top 15 states by property count
        - Bottom-right: Properties over time (if year column exists)
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe with property_type, ownership, state columns
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top-left: Property type distribution
        self._plot_property_types(df, ax=axes[0, 0])
        
        # Top-right: Ownership breakdown
        self._plot_ownership_pie(df, ax=axes[0, 1])
        
        # Bottom-left: Top states
        self._plot_top_states(df, ax=axes[1, 0], top_n=15)
        
        # Bottom-right: Properties by year (if year exists)
        if 'year' in df.columns:
            self._plot_properties_by_year(df, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Year column not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def _plot_property_types(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot property type distribution."""
        if 'property_type' not in df.columns:
            ax.text(0.5, 0.5, 'property_type column not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        property_counts = df['property_type'].value_counts()
        property_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Properties', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Distribution of Property Types', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
    
    def _plot_ownership_pie(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot ownership breakdown pie chart."""
        if 'ownership' not in df.columns:
            ax.text(0.5, 0.5, 'ownership column not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        ownership_counts = df['ownership'].value_counts()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6', '#f39c12', '#9b59b6']
        ax.pie(ownership_counts, labels=ownership_counts.index, autopct='%1.1f%%', 
               colors=colors[:len(ownership_counts)], startangle=90)
        ax.set_title('Ownership Distribution', fontsize=12, fontweight='bold')
    
    def _plot_top_states(self, df: pd.DataFrame, ax: plt.Axes, top_n: int = 15):
        """Plot top states by property count."""
        if 'state' not in df.columns:
            ax.text(0.5, 0.5, 'state column not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        state_counts = df['state'].value_counts().head(top_n)
        bars = ax.bar(range(len(state_counts)), state_counts.values, 
                      color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(state_counts)))
        ax.set_xticklabels(state_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Properties', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} States by Property Count', fontsize=12, fontweight='bold')
    
    def _plot_properties_by_year(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot properties over time."""
        yearly = df.groupby('year').size()
        ax.bar(yearly.index.astype(int), yearly.values, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Properties', fontsize=11, fontweight='bold')
        ax.set_title('Properties Reported Over Time', fontsize=12, fontweight='bold')
    
    def plot_property_type_distribution(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot property type and ownership distribution side by side.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Property type counts
        self._plot_property_types(df, ax=axes[0])
        
        # Right: Ownership breakdown
        self._plot_ownership_pie(df, ax=axes[1])
        
        plt.tight_layout()
        return fig
    
    def plot_top_states(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot top states by property count.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        top_n : int
            Number of top states to show
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'state' not in df.columns:
            ax.text(0.5, 0.5, 'state column not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        state_counts = df['state'].value_counts().head(top_n)
        bars = ax.bar(range(len(state_counts)), state_counts.values, 
                      color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(state_counts)))
        ax.set_xticklabels(state_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Properties', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} States by Property Count', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, state_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(state_counts) * 0.01, 
                    f'{val:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_ownership_by_type(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot stacked bar chart of ownership breakdown by property type.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'property_type' not in df.columns or 'ownership' not in df.columns:
            ax.text(0.5, 0.5, 'Required columns not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        ownership_by_type = pd.crosstab(df['property_type'], df['ownership'], normalize='index') * 100
        
        ownership_by_type.plot(kind='barh', stacked=True, ax=ax, colormap='Set2', edgecolor='white')
        ax.set_xlabel('Percentage', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Ownership Breakdown by Property Type', fontsize=12, fontweight='bold')
        ax.legend(title='Ownership', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def plot_property_size_distribution(
        self,
        df: pd.DataFrame,
        size_col: str = 'size_sqft',
        type_col: str = 'property_type',
        min_samples: int = 10,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot property size distribution with log scale.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        size_col : str
            Column name for property size
        type_col : str
            Column name for property type
        min_samples : int
            Minimum samples per type for box plot
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if size_col not in df.columns:
            axes[0].text(0.5, 0.5, f'{size_col} column not available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, f'{size_col} column not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            return fig
        
        # Filter to valid size data
        df_with_size = df[df[size_col].notna() & (df[size_col] > 0)].copy()
        
        if len(df_with_size) == 0:
            axes[0].text(0.5, 0.5, 'No valid size data available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No valid size data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            return fig
        
        # Left: Histogram with logarithmic bins (fixed)
        size_data = df_with_size[size_col].values
        
        # Create logarithmic bins for proper log-scale histogram
        log_min = np.floor(np.log10(size_data.min()))
        log_max = np.ceil(np.log10(size_data.max()))
        bins = np.logspace(log_min, log_max, 50)
        
        axes[0].hist(size_data, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Size (square feet)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Property Size Distribution (n={len(df_with_size):,})', fontsize=12, fontweight='bold')
        axes[0].set_xscale('log')
        
        # Add mean/median lines
        mean_size = size_data.mean()
        median_size = np.median(size_data)
        axes[0].axvline(mean_size, color='coral', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_size:,.0f}')
        axes[0].axvline(median_size, color='gold', linestyle='--', linewidth=2, 
                        label=f'Median: {median_size:,.0f}')
        axes[0].legend(loc='upper right')
        
        # Right: Box plot of size by property type
        if type_col in df.columns:
            size_by_type = df_with_size.groupby(type_col)[size_col].apply(list).to_dict()
            types_with_data = [k for k, v in size_by_type.items() if len(v) >= min_samples]
            
            if types_with_data:
                data_for_box = [size_by_type[t] for t in types_with_data]
                bp = axes[1].boxplot(data_for_box, vert=True, patch_artist=True)
                axes[1].set_xticklabels(types_with_data, rotation=45, ha='right')
                axes[1].set_ylabel('Size (square feet)', fontsize=11, fontweight='bold')
                axes[1].set_title('Property Size by Type', fontsize=12, fontweight='bold')
                axes[1].set_yscale('log')
                
                # Color the boxes
                colors = sns.color_palette('Set2', len(types_with_data))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            else:
                axes[1].text(0.5, 0.5, 'Insufficient data for box plots', 
                            ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, f'{type_col} column not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_headquarters_analysis(
        self,
        df: pd.DataFrame,
        hq_col: str = 'is_headquarters',
        state_col: str = 'state',
        city_col: str = 'city',
        top_n: int = 15,
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Analyze headquarters locations by state and city.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        hq_col : str
            Column indicating headquarters
        state_col : str
            State column name
        city_col : str
            City column name
        top_n : int
            Number of top locations to show
        figsize : tuple
            Figure size
        
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if hq_col not in df.columns:
            axes[0].text(0.5, 0.5, f'{hq_col} column not available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, f'{hq_col} column not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            return fig
        
        # Filter to headquarters
        hq_df = df[df[hq_col] == True]
        
        if len(hq_df) == 0:
            axes[0].text(0.5, 0.5, 'No headquarters found', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No headquarters found', 
                        ha='center', va='center', transform=axes[1].transAxes)
            return fig
        
        # Left: Top states with headquarters
        if state_col in df.columns:
            hq_states = hq_df[state_col].value_counts().head(top_n)
            axes[0].barh(range(len(hq_states)), hq_states.values, color='coral', edgecolor='black')
            axes[0].set_yticks(range(len(hq_states)))
            axes[0].set_yticklabels(hq_states.index)
            axes[0].set_xlabel('Number of Headquarters', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Top {top_n} States for HQs (n={len(hq_df):,})', fontsize=12, fontweight='bold')
            axes[0].invert_yaxis()
        else:
            axes[0].text(0.5, 0.5, f'{state_col} column not available', 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        # Right: Top cities with headquarters
        if city_col in df.columns:
            hq_cities = hq_df[city_col].value_counts().head(top_n)
            axes[1].barh(range(len(hq_cities)), hq_cities.values, color='teal', edgecolor='black')
            axes[1].set_yticks(range(len(hq_cities)))
            axes[1].set_yticklabels(hq_cities.index)
            axes[1].set_xlabel('Number of Headquarters', fontsize=11, fontweight='bold')
            axes[1].set_title(f'Top {top_n} Cities for Headquarters', fontsize=12, fontweight='bold')
            axes[1].invert_yaxis()
        else:
            axes[1].text(0.5, 0.5, f'{city_col} column not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics table for location data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Location dataframe
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        rows = [
            {'metric': 'Total properties', 'value': len(df)},
            {'metric': 'Unique firms', 'value': df['cik'].nunique() if 'cik' in df.columns else 'N/A'},
            {'metric': 'Unique states', 'value': df['state'].nunique() if 'state' in df.columns else 'N/A'},
            {'metric': 'Unique cities', 'value': df['city'].nunique() if 'city' in df.columns else 'N/A'},
        ]
        
        if 'property_type' in df.columns:
            rows.append({'metric': 'Property types', 'value': df['property_type'].nunique()})
        
        if 'is_headquarters' in df.columns:
            hq_count = df[df['is_headquarters'] == True].shape[0]
            rows.append({'metric': 'Headquarters', 'value': hq_count})
        
        if 'size_sqft' in df.columns:
            df_valid = df[df['size_sqft'].notna() & (df['size_sqft'] > 0)]
            rows.append({'metric': 'Properties with size', 'value': len(df_valid)})
            if len(df_valid) > 0:
                rows.append({'metric': 'Mean size (sqft)', 'value': f"{df_valid['size_sqft'].mean():,.0f}"})
                rows.append({'metric': 'Median size (sqft)', 'value': f"{df_valid['size_sqft'].median():,.0f}"})
        
        if 'year' in df.columns:
            rows.append({'metric': 'Years', 'value': f"{df['year'].min()} - {df['year'].max()}"})
        
        return pd.DataFrame(rows)
