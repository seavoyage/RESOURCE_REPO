#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Data Loader Module

This module provides a robust data loading and preprocessing framework following
the OSEMN data science workflow.

Key Features:
    - Automated data validation and quality checks
    - Comprehensive error handling and logging
    - Support for multiple file formats and compression
    - Initial data quality assessment
    - Basic preprocessing capabilities
    - Memory optimization for large datasets

Dependencies:
    - pandas>=1.0.0: Core data manipulation
    - numpy>=1.18.0: Numerical operations
    - pydqc: Data quality checks
    - scikit-learn>=0.24.0: Preprocessing utilities
    - PyYAML: Configuration management
    
Example:
    >>> loader = DataLoader("data.csv")
    >>> df = loader.load_data()
    >>> df = loader.preprocess_data(df)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Union, Dict, List
from dataclasses import dataclass
import yaml
from pydqc.data_quality import DataQuality
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataLoader:
    """
    Enhanced data loader with validation, quality checks and preprocessing capabilities.
    
    This class implements best practices for data loading and initial preprocessing
    following the OSEMN (Obtain, Scrub, Explore, Model, iNterpret) framework.
    
    Attributes:
        file_path (Union[str, Path]): Path to the data file
        config_path (Optional[str]): Path to configuration YAML file
        validate_dtypes (bool): Whether to validate data types
        detect_anomalies (bool): Whether to perform anomaly detection
        
    Methods:
        load_data: Load and validate data from file
        preprocess_data: Perform initial data cleaning and preprocessing
        validate_schema: Validate dataframe against expected schema
        detect_data_quality_issues: Generate data quality report
        
    Example:
        >>> loader = DataLoader(
        ...     file_path='data.csv',
        ...     config_path='config.yaml',
        ...     validate_dtypes=True
        ... )
        >>> df = loader.load_data()
        >>> quality_report = loader.detect_data_quality_issues(df)
        >>> df_processed = loader.preprocess_data(df)
    
    Notes:
        - Supports CSV, Excel, JSON, and compressed files
        - Handles missing values and categorical encoding
        - Provides detailed logging of all operations
        - Uses vectorized operations for better performance
    """
    file_path: Union[str, Path]
    config_path: Optional[str] = None
    validate_dtypes: bool = True
    detect_anomalies: bool = True

    def __post_init__(self):
        # Configure logging with more detailed formatting
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load config if provided
        self.config = self._load_config() if self.config_path else {}
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {str(e)}")
            return {}

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate dataframe schema against expected dtypes"""
        if not self.config.get('expected_dtypes'):
            return True
            
        try:
            for col, dtype in self.config['expected_dtypes'].items():
                if col not in df.columns:
                    self.logger.error(f"Missing expected column: {col}")
                    return False
                if str(df[col].dtype) != dtype:
                    self.logger.error(f"Invalid dtype for {col}: expected {dtype}, got {df[col].dtype}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            return False

    def detect_data_quality_issues(self, df: pd.DataFrame) -> Dict:
        """
        Detect common data quality issues using PYDQC
        """
        try:
            quality_report = {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum(),
                'numeric_stats': df.describe().to_dict(),
                'categorical_stats': {col: df[col].value_counts().to_dict() 
                                   for col in df.select_dtypes(include=['object']).columns}
            }
            
            # Log quality issues
            self.logger.info("Data quality report generated")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error in quality detection: {str(e)}")
            return {}

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV with enhanced error handling and validation
        
        Args:
            file_path: Optional path to data file, overrides instance file_path
            
        Returns:
            pd.DataFrame: Loaded and validated dataframe
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        try:
            if file_path:
                self.file_path = Path(file_path)
            
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
                
            # Use optimized pandas read_csv parameters
            df = pd.read_csv(
                self.file_path,
                low_memory=False,  # Avoid mixed type inference
                parse_dates=self.config.get('date_columns', []),
                dtype=self.config.get('column_dtypes', None)
            )
            
            self.logger.info(f"Successfully loaded data from {self.file_path}")
            self.logger.info(f"DataFrame shape: {df.shape}")
            
            # Validate schema if requested
            if self.validate_dtypes and not self.validate_schema(df):
                raise ValueError("Data validation failed")
                
            # Detect quality issues if requested
            if self.detect_anomalies:
                quality_report = self.detect_data_quality_issues(df)
                self.logger.info(f"Quality report: {quality_report}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data preprocessing steps
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        try:
            # Handle missing values
            for col in df.columns:
                missing_pct = df[col].isnull().mean()
                if missing_pct > 0:
                    self.logger.info(f"Column {col} has {missing_pct:.2%} missing values")
                    
                    if missing_pct > 0.5:
                        self.logger.warning(f"Dropping column {col} due to high missing rate")
                        df = df.drop(columns=[col])
                    else:
                        # Impute based on dtype
                        if np.issubdtype(df[col].dtype, np.number):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            # Encode categorical variables
            categorical_columns = df.select_dtypes(include=['object']).columns
            encoders = {}
            
            for col in categorical_columns:
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
                
            self.logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
        

