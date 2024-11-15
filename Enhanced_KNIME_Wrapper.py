#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import knime.extension as knext
import pandas as pd
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler, QuantileTransformer, PowerTransformer,
    OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path
import yaml

@dataclass
class PipelineConfig:
    """Configuration for data science pipeline"""
    random_state: int = 42
    categorical_threshold: int = 10
    max_categories: int = 50
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.5
    n_bins: int = 10

@dataclass
class PipelineMetrics:
    """Container for pipeline metrics"""
    data_quality_score: float
    missing_rates: Dict[str, float]
    feature_importance: Dict[str, float]
    transformation_logs: List[Dict]

def setup_logging(log_file: str = 'pipeline.log') -> logging.Logger:
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

def create_preprocessing_pipeline(
    categorical_columns: List[str],
    numerical_columns: List[str],
    config: PipelineConfig
) -> Pipeline:
    """
    Create scikit-learn pipeline for data preprocessing
    
    Args:
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        config: Pipeline configuration
        
    Returns:
        Preprocessing pipeline
    """
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        verbose_feature_names_out=False
    )

    return Pipeline([
        ('preprocessor', preprocessor)
    ])

@knext.node(
    name="Data Science Pipeline",
    node_type=knext.NodeType.LEARNER,
    icon_path="icon.png",
    category="/Community/Data Science"
)
class DataSciencePipelineNode:
    """
    KNIME node wrapper for comprehensive data science pipeline
    
    Features:
    - Automated data quality assessment
    - Intelligent feature preprocessing
    - Comprehensive logging and metrics
    - Configurable pipeline parameters
    """
    
    # Node parameters
    file_path = knext.StringParameter(
        "Input File Path",
        "Path to the input data file",
        "data/input.csv"
    )
    
    config_path = knext.StringParameter(
        "Configuration Path",
        "Path to pipeline configuration YAML",
        "config/pipeline_config.yaml",
        optional=True
    )
    
    def configure(self, configure_context, input_schema):
        """Validate configuration and input schema"""
        try:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Input file not found: {self.file_path}")
                
            if self.config_path and not Path(self.config_path).exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            return None
            
        except Exception as e:
            configure_context.logger.error(f"Configuration error: {str(e)}")
            raise
    
    def execute(self, exec_context, input_data) -> knext.Table:
        """
        Execute data science pipeline
        
        Args:
            exec_context: KNIME execution context
            input_data: Input data table
            
        Returns:
            Processed data table with metrics
        """
        try:
            # Setup logging and configuration
            logger = setup_logging()
            config = self._load_config()
            logger.info("Pipeline started with configuration loaded")
            
            # Load and validate data
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Identify column types
            categorical_columns = df.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            numerical_columns = df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            # Create and fit preprocessing pipeline
            pipeline = create_preprocessing_pipeline(
                categorical_columns,
                numerical_columns,
                config
            )
            
            # Transform data
            df_transformed = pd.DataFrame(
                pipeline.fit_transform(df),
                columns=pipeline.get_feature_names_out()
            )
            
            # Calculate pipeline metrics
            metrics = PipelineMetrics(
                data_quality_score=self._calculate_quality_score(df),
                missing_rates=df.isnull().mean().to_dict(),
                feature_importance=self._calculate_feature_importance(df),
                transformation_logs=self._get_transformation_logs(pipeline)
            )
            
            # Log metrics
            logger.info(f"Pipeline metrics: {metrics}")
            
            return knext.Table.from_pandas(df_transformed)
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            raise
    
    def _load_config(self) -> PipelineConfig:
        """Load pipeline configuration from YAML"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return PipelineConfig(**config_dict)
        return PipelineConfig()
    
    @staticmethod
    def _calculate_quality_score(df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        missing_penalty = df.isnull().mean().mean()
        unique_ratio = df.nunique().mean() / len(df)
        return 1 - (missing_penalty + (1 - unique_ratio)) / 2
    
    @staticmethod
    def _calculate_feature_importance(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores"""
        return {
            col: df[col].nunique() / len(df)
            for col in df.columns
        }
    
    @staticmethod
    def _get_transformation_logs(pipeline: Pipeline) -> List[Dict]:
        """Extract transformation logs from pipeline"""
        return [
            {
                'step': name,
                'transformer': str(transformer),
                'parameters': transformer.get_params()
            }
            for name, transformer in pipeline.named_steps.items()
        ]

