#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pydqc.data_quality import DQC
from openrefine import refine
from sklearn.preprocessing import (
    RobustScaler, QuantileTransformer, PowerTransformer, LabelEncoder
)
from sklearn.impute import SimpleImputer
import optbinning
from scipy import stats
import logging
from pathlib import Path
import yaml

@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""
    missing_rates: Dict[str, float]
    outlier_counts: Dict[str, int]
    skewness: Dict[str, float]
    unique_counts: Dict[str, int]
    correlation_matrix: pd.DataFrame
    data_types: Dict[str, str]

@dataclass
class TransformationLog:
    """Track applied transformations"""
    column: str
    transformation: str
    parameters: Dict
    timestamp: str

def setup_logging(log_file: str = 'data_quality.log') -> logging.Logger:
    """Configure logging with file and console handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_distributions(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze numerical distributions using vectorized operations
    
    Returns:
        Dictionary containing distribution metrics for each numerical column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return {
        col: {
            'skewness': float(stats.skew(df[col].dropna())),
            'kurtosis': float(stats.kurtosis(df[col].dropna())),
            'outliers': len(df[df[col].abs() > df[col].std() * 3]),
            'missing_rate': df[col].isnull().mean()
        }
        for col in numeric_cols
    }

def select_transformer(
    data: np.ndarray,
    n_samples: int,
    has_outliers: bool,
    is_positive: bool
) -> Tuple[str, object]:
    """
    Select appropriate transformer based on data characteristics
    """
    if has_outliers:
        if n_samples > 1000:
            return 'quantile', QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(n_samples // 2, 1000)
            )
        return 'robust', RobustScaler()
    
    if is_positive:
        return 'box-cox', PowerTransformer(method='box-cox')
    return 'yeo-johnson', PowerTransformer(method='yeo-johnson')

def process_missing_data(
    df: pd.DataFrame,
    strategy: str = 'median',
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Handle missing data using vectorized operations
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        threshold: Maximum allowed missing rate before column dropping
        
    Returns:
        Processed DataFrame and imputation summary
    """
    # Analyze missing patterns
    missing_rates = df.isnull().mean()
    high_missing = missing_rates[missing_rates > threshold].index
    
    # Drop columns with too many missing values
    df_cleaned = df.drop(columns=high_missing)
    
    # Initialize imputers
    num_imputer = SimpleImputer(strategy=strategy)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Separate numerical and categorical columns
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    
    # Apply imputation
    if len(num_cols) > 0:
        df_cleaned[num_cols] = num_imputer.fit_transform(df_cleaned[num_cols])
    if len(cat_cols) > 0:
        df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])
    
    imputation_summary = {
        'dropped_columns': list(high_missing),
        'imputed_numerical': list(num_cols),
        'imputed_categorical': list(cat_cols)
    }
    
    return df_cleaned, imputation_summary

def normalize_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[TransformationLog]]:
    """
    Normalize numerical features using appropriate transformations
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        Normalized DataFrame and transformation logs
    """
    df_normalized = df.copy()
    transformation_logs = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in numeric_cols:
        data = df[col].values.reshape(-1, 1)
        n_samples = len(data)
        
        # Analyze distribution
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        has_outliers = any(z_scores > 3)
        is_positive = all(df[col].dropna() > 0)
        
        # Select and apply transformer
        transform_name, transformer = select_transformer(
            data, n_samples, has_outliers, is_positive
        )
        
        df_normalized[col] = transformer.fit_transform(data)
        
        # Log transformation
        transformation_logs.append(
            TransformationLog(
                column=col,
                transformation=transform_name,
                parameters=transformer.get_params(),
                timestamp=pd.Timestamp.now().isoformat()
            )
        )
    
    return df_normalized, transformation_logs

class DataQualityAnalyzer:
    """
    Functional interface for data quality analysis and transformation pipeline
    following OSEMN framework
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional configuration"""
        self.logger = setup_logging()
        self.config = self._load_config(config_path) if config_path else {}
        self.transformation_history = []
    
    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_quality(
        self,
        df: pd.DataFrame,
        generate_report: bool = True
    ) -> DataQualityMetrics:
        """
        Comprehensive data quality analysis
        """
        try:
            # Generate PYDQC report if requested
            if generate_report:
                dqc = DQC(df)
                dqc.get_report()
            
            # Calculate quality metrics
            metrics = DataQualityMetrics(
                missing_rates=df.isnull().mean().to_dict(),
                outlier_counts={
                    col: sum(np.abs(stats.zscore(df[col].dropna())) > 3)
                    for col in df.select_dtypes(include=[np.number]).columns
                },
                skewness=df.select_dtypes(include=[np.number]).skew().to_dict(),
                unique_counts=df.nunique().to_dict(),
                correlation_matrix=df.corr(),
                data_types=df.dtypes.astype(str).to_dict()
            )
            
            self.logger.info("Data quality analysis completed")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in quality analysis: {str(e)}")
            raise

