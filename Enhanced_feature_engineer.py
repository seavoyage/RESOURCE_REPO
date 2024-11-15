#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, PolynomialFeatures,
    RobustScaler, QuantileTransformer
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from featuretools import dfs, EntitySet
import optbinning
from optbinning import OptimalBinning
import logging
from pathlib import Path

@dataclass
class FeatureMetadata:
    """Container for feature metadata"""
    name: str
    type: str
    unique_count: int
    missing_rate: float
    importance_score: Optional[float] = None
    encoding_map: Optional[Dict] = None

@dataclass
class TransformationConfig:
    """Feature transformation configuration"""
    categorical_threshold: int = 10
    max_categories: int = 50
    polynomial_degree: int = 2
    n_bins: int = 10
    correlation_threshold: float = 0.95

def setup_logging(log_file: str = 'feature_engineering.log') -> logging.Logger:
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

def analyze_features(df: pd.DataFrame) -> Dict[str, FeatureMetadata]:
    """
    Analyze features and generate metadata
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping feature names to their metadata
    """
    metadata = {}
    
    for col in df.columns:
        metadata[col] = FeatureMetadata(
            name=col,
            type='numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
            unique_count=df[col].nunique(),
            missing_rate=df[col].isnull().mean()
        )
    
    return metadata

def encode_categorical_features(
    df: pd.DataFrame,
    metadata: Dict[str, FeatureMetadata],
    config: TransformationConfig
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical features using appropriate methods
    
    Args:
        df: Input DataFrame
        metadata: Feature metadata
        config: Transformation configuration
        
    Returns:
        Transformed DataFrame and encoding mappings
    """
    df_encoded = df.copy()
    encoding_maps = {}
    
    for col, meta in metadata.items():
        if meta.type == 'categorical':
            # Determine encoding method
            if meta.unique_count <= config.categorical_threshold:
                # Use label encoding for low cardinality ordinal features
                encoder = LabelEncoder()
                df_encoded[f'{col}_encoded'] = encoder.fit_transform(df[col].fillna('missing'))
                encoding_maps[col] = {
                    'method': 'label',
                    'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                }
            else:
                # Use one-hot encoding with handling for high cardinality
                if meta.unique_count > config.max_categories:
                    # Bin rare categories
                    value_counts = df[col].value_counts()
                    top_categories = value_counts.nlargest(config.max_categories).index
                    df[col] = df[col].where(df[col].isin(top_categories), 'Other')
                
                # Apply one-hot encoding
                onehot = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, onehot], axis=1)
                encoding_maps[col] = {
                    'method': 'onehot',
                    'columns': onehot.columns.tolist()
                }
            
            # Drop original column
            df_encoded.drop(columns=[col], inplace=True)
    
    return df_encoded, encoding_maps

def create_polynomial_features(
    df: pd.DataFrame,
    metadata: Dict[str, FeatureMetadata],
    config: TransformationConfig
) -> pd.DataFrame:
    """
    Create polynomial features for numeric columns
    
    Args:
        df: Input DataFrame
        metadata: Feature metadata
        config: Transformation configuration
        
    Returns:
        DataFrame with polynomial features
    """
    numeric_cols = [col for col, meta in metadata.items() 
                   if meta.type == 'numeric']
    
    if not numeric_cols:
        return df
    
    poly = PolynomialFeatures(
        degree=config.polynomial_degree,
        include_bias=False
    )
    
    poly_features = poly.fit_transform(df[numeric_cols])
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    return pd.concat([
        df,
        pd.DataFrame(poly_features[:, len(numeric_cols):], 
                    columns=feature_names[len(numeric_cols):])
    ], axis=1)

def optimize_binning(
    df: pd.DataFrame,
    target: pd.Series,
    metadata: Dict[str, FeatureMetadata],
    config: TransformationConfig
) -> pd.DataFrame:
    """
    Optimize feature binning using OptBinning
    
    Args:
        df: Input DataFrame
        target: Target variable
        metadata: Feature metadata
        config: Transformation configuration
        
    Returns:
        DataFrame with optimized binned features
    """
    df_binned = df.copy()
    
    for col, meta in metadata.items():
        if meta.type == 'numeric':
            optb = OptimalBinning(
                name=col,
                dtype='numerical',
                solver='cp',
                max_n_bins=config.n_bins
            )
            optb.fit(df[col], target)
            df_binned[f'{col}_binned'] = optb.transform(df[col], metric='woe')
    
    return df_binned

class FeatureEngineer:
    """
    Automated feature engineering pipeline with comprehensive tracking
    and optimization capabilities
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize feature engineer with configuration
        
        Args:
            config_path: Path to YAML configuration file
            random_state: Random seed for reproducibility
        """
        self.logger = setup_logging()
        self.config = TransformationConfig()
        self.random_state = random_state
        np.random.seed(random_state)
        
        if config_path:
            self._load_config(config_path)
    
    def transform_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply comprehensive feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target: Optional target variable for supervised transformations
            
        Returns:
            Transformed DataFrame and transformation metadata
        """
        try:
            # Analyze features
            metadata = analyze_features(df)
            self.logger.info(f"Analyzed {len(metadata)} features")
            
            # Encode categorical features
            df_encoded, encoding_maps = encode_categorical_features(
                df, metadata, self.config
            )
            self.logger.info("Completed categorical encoding")
            
            # Create polynomial features
            df_poly = create_polynomial_features(
                df_encoded, metadata, self.config
            )
            self.logger.info("Created polynomial features")
            
            # Optimize binning if target is provided
            if target is not None:
                df_final = optimize_binning(
                    df_poly, target, metadata, self.config
                )
                self.logger.info("Completed optimal binning")
            else:
                df_final = df_poly
            
            # Track transformations
            transformation_metadata = {
                'feature_metadata': metadata,
                'encoding_maps': encoding_maps,
                'n_features_original': len(df.columns),
                'n_features_transformed': len(df_final.columns)
            }
            
            return df_final, transformation_metadata
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise

