from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
import yaml
import logging
from ..agents.eda_agent import EDAAgent
from ..agents.preprocessing_agent import PreprocessingAgent
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class DataPipeline:
    """
    Main class for the autonomous data pipeline builder.
    Coordinates the EDA, preprocessing, and pipeline generation process.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataPipeline with optional configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.data: Optional[pd.DataFrame] = None
        self.eda_agent = EDAAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.pipeline: Optional[Pipeline] = None
        self.eda_results: Dict[str, Any] = {}
        self.preprocessing_recommendations: Dict[str, Any] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> None:
        """
        Load data from various sources (CSV, DataFrame, etc.).
        
        Args:
            data_source: Path to data file or DataFrame
            **kwargs: Additional arguments for data loading
        """
        try:
            if isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            elif isinstance(data_source, str):
                file_extension = Path(data_source).suffix.lower()
                if file_extension == '.csv':
                    self.data = pd.read_csv(data_source, **kwargs)
                elif file_extension == '.json':
                    self.data = pd.read_json(data_source, **kwargs)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            else:
                raise ValueError("data_source must be a DataFrame or path to a file")
            
            logger.info(f"Successfully loaded data with shape {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run autonomous EDA on the loaded data.
        
        Returns:
            Dictionary containing EDA results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            self.eda_results = self.eda_agent.analyze(self.data)
            logger.info("Completed EDA analysis")
            return self.eda_results
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get preprocessing recommendations based on EDA results.
        
        Returns:
            Dictionary containing preprocessing recommendations
        """
        if not self.eda_results:
            raise ValueError("No EDA results available. Call analyze() first.")
        
        try:
            self.preprocessing_recommendations = (
                self.preprocessing_agent.generate_recommendations(
                    self.data, 
                    self.eda_results
                )
            )
            logger.info("Generated preprocessing recommendations")
            return self.preprocessing_recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def generate(self, custom_steps: Optional[List[Dict[str, Any]]] = None) -> Pipeline:
        """
        Generate a scikit-learn pipeline based on recommendations.
        
        Args:
            custom_steps: Optional list of custom preprocessing steps
            
        Returns:
            sklearn.pipeline.Pipeline object
        """
        if not self.preprocessing_recommendations:
            raise ValueError("No preprocessing recommendations available. Call get_recommendations() first.")
        
        try:
            self.pipeline = self.preprocessing_agent.build_pipeline(
                self.preprocessing_recommendations,
                custom_steps
            )
            logger.info("Generated sklearn pipeline")
            return self.pipeline
        except Exception as e:
            logger.error(f"Error generating pipeline: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save the generated pipeline to disk.
        
        Args:
            path: Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("No pipeline generated. Call generate() first.")
        
        try:
            joblib.dump(self.pipeline, path)
            logger.info(f"Saved pipeline to {path}")
        except Exception as e:
            logger.error(f"Error saving pipeline: {str(e)}")
            raise
    
    def load_pipeline(self, path: str) -> None:
        """
        Load a previously saved pipeline.
        
        Args:
            path: Path to the saved pipeline
        """
        try:
            self.pipeline = joblib.load(path)
            logger.info(f"Loaded pipeline from {path}")
        except Exception as e:
            logger.error(f"Error loading pipeline: {str(e)}")
            raise 