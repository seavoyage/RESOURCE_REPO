#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""
    missing_rates: Dict[str, float]
    outlier_counts: Dict[str, int]
    skewness: Dict[str, float]
    unique_counts: Dict[str, int]

class DataAnalyzer:
    """
    Automated data analysis and cleaning with parallel processing capabilities.
    
    Implements OSEMN framework:
    - Obtain: Load and validate data
    - Scrub: Clean and preprocess
    - Explore: Analysis and visualization
    - Model: Prepare for modeling
    - iNterpret: Generate insights
    """
    
    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        """
        Initialize analyzer with dataset and configuration.
        
        Args:
            df: Input DataFrame
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Identify column types
        self._identify_column_types()
        
        # Initialize transformers
        self.scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal',
            random_state=random_state
        )
        
    def _identify_column_types(self) -> None:
        """Categorize columns by data type"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        
    def analyze_data_quality(self) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            DataQualityMetrics object containing quality metrics
        """
        # Calculate missing value rates
        missing_rates = self.df.isnull().mean().to_dict()
        
        # Detect outliers using IQR method
        outlier_counts = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                       (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts[col] = outliers
            
        # Calculate skewness
        skewness = self.df[self.numeric_cols].skew().to_dict()
        
        # Count unique values
        unique_counts = self.df.nunique().to_dict()
        
        return DataQualityMetrics(
            missing_rates=missing_rates,
            outlier_counts=outlier_counts,
            skewness=skewness,
            unique_counts=unique_counts
        )

    def generate_visualizations(self, output_dir: str = 'plots') -> None:
        """
        Generate comprehensive visualization suite using Seaborn.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set consistent style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        def plot_numeric_distribution(col: str) -> None:
            """Plot distribution for numeric column"""
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {col}')
            
            # Box plot
            sns.boxplot(data=self.df, x=col, ax=ax2)
            ax2.set_title(f'Box Plot of {col}')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{col}_analysis.png')
            plt.close()
            
        # Generate plots in parallel
        with ProcessPoolExecutor() as executor:
            executor.map(plot_numeric_distribution, self.numeric_cols)
            
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.df[self.numeric_cols].corr(),
            annot=True,
            cmap='coolwarm',
            center=0
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()

    def prepare_for_modeling(
        self,
        handle_missing: bool = True,
        scale_features: bool = True,
        transform_skewed: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for modeling following best practices.
        
        Args:
            handle_missing: Whether to impute missing values
            scale_features: Whether to scale numeric features
            transform_skewed: Whether to transform skewed distributions
            
        Returns:
            Processed DataFrame ready for modeling
        """
        df_processed = self.df.copy()
        
        if handle_missing:
            # Numeric imputation
            num_imputer = SimpleImputer(strategy='median')
            df_processed[self.numeric_cols] = num_imputer.fit_transform(
                df_processed[self.numeric_cols]
            )
            
            # Categorical imputation
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[self.categorical_cols] = cat_imputer.fit_transform(
                df_processed[self.categorical_cols]
            )
            
        if scale_features:
            df_processed[self.numeric_cols] = self.scaler.fit_transform(
                df_processed[self.numeric_cols]
            )
            
        if transform_skewed:
            # Transform highly skewed features
            skewed_features = df_processed[self.numeric_cols].apply(lambda x: abs(x.skew()))
            highly_skewed = skewed_features[skewed_features > 1.0].index
            
            if len(highly_skewed) > 0:
                df_processed[highly_skewed] = self.quantile_transformer.fit_transform(
                    df_processed[highly_skewed]
                )
                
        return df_processed

