from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PreprocessingStep(BaseModel):
    """Pydantic model for a preprocessing step configuration."""
    name: str = Field(description="Name of the preprocessing step")
    transformer: str = Field(description="Transformer class to use")
    columns: List[str] = Field(description="Columns to apply the transformer to")
    params: Dict[str, Any] = Field(description="Parameters for the transformer")

class PreprocessingRecommendations(BaseModel):
    """Pydantic model for preprocessing recommendations."""
    steps: List[PreprocessingStep] = Field(description="List of preprocessing steps")
    reasoning: List[str] = Field(description="Reasoning for each step")

class PreprocessingAgent:
    """
    Agent for generating and building preprocessing pipelines.
    Uses LLM for intelligent preprocessing decisions.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        """
        Initialize the preprocessing agent.
        
        Args:
            model_name: Name of the LLM model to use (default: gpt-3.5-turbo)
            temperature: Temperature for LLM responses
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.output_parser = PydanticOutputParser(pydantic_object=PreprocessingRecommendations)
        
        # Available transformers
        self.transformers = {
            "standard_scaler": StandardScaler,
            "minmax_scaler": MinMaxScaler,
            "robust_scaler": RobustScaler,
            "onehot_encoder": OneHotEncoder,
            "label_encoder": LabelEncoder,
            "ordinal_encoder": OrdinalEncoder,
            "simple_imputer": SimpleImputer,
            "knn_imputer": KNNImputer,
            "drop_constant": DropConstantFeatures,
            "drop_duplicate": DropDuplicateFeatures,
            "smart_correlation": SmartCorrelatedSelection
        }
    
    def generate_recommendations(
        self,
        data: pd.DataFrame,
        eda_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate preprocessing recommendations based on EDA results.
        
        Args:
            data: Input DataFrame
            eda_results: Results from EDA analysis
            
        Returns:
            Dictionary containing preprocessing recommendations
        """
        try:
            # Get LLM recommendations
            recommendations = self._get_llm_recommendations(data, eda_results)
            
            # Validate recommendations
            validated_steps = self._validate_steps(recommendations.steps, data)
            
            return {
                "steps": [step.dict() for step in validated_steps],
                "reasoning": recommendations.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def build_pipeline(
        self,
        recommendations: Dict[str, Any],
        custom_steps: Optional[List[Dict[str, Any]]] = None
    ) -> Pipeline:
        """
        Build a scikit-learn pipeline from recommendations.
        
        Args:
            recommendations: Preprocessing recommendations
            custom_steps: Optional custom preprocessing steps
            
        Returns:
            sklearn.pipeline.Pipeline object
        """
        try:
            steps = []
            
            # Add feature selection steps first
            selection_steps = [
                step for step in recommendations["steps"]
                if step["transformer"] in [
                    "drop_constant",
                    "drop_duplicate",
                    "smart_correlation"
                ]
            ]
            for step in selection_steps:
                transformer_class = self.transformers[step["transformer"]]
                transformer = transformer_class(**step["params"])
                steps.append((step["name"], transformer))
            
            # Group remaining transformers by type
            numeric_transformers = []
            categorical_transformers = []
            
            for step in recommendations["steps"]:
                if step["transformer"] not in [t["transformer"] for t in selection_steps]:
                    transformer_class = self.transformers[step["transformer"]]
                    transformer = transformer_class(**step["params"])
                    
                    if step["transformer"] in [
                        "standard_scaler",
                        "minmax_scaler",
                        "robust_scaler",
                        "simple_imputer",
                        "knn_imputer"
                    ]:
                        numeric_transformers.append((
                            step["name"],
                            transformer,
                            step["columns"]
                        ))
                    else:
                        categorical_transformers.append((
                            step["name"],
                            transformer,
                            step["columns"]
                        ))
            
            # Create column transformers
            transformers = []
            
            if numeric_transformers:
                numeric_pipeline = Pipeline([
                    (name, trans) for name, trans, _ in numeric_transformers
                ])
                numeric_columns = numeric_transformers[0][2]  # Use columns from first transformer
                transformers.append(("numeric", numeric_pipeline, numeric_columns))
            
            if categorical_transformers:
                categorical_pipeline = Pipeline([
                    (name, trans) for name, trans, _ in categorical_transformers
                ])
                categorical_columns = categorical_transformers[0][2]
                transformers.append(("categorical", categorical_pipeline, categorical_columns))
            
            # Add column transformer to steps
            if transformers:
                steps.append((
                    "column_transformer",
                    ColumnTransformer(transformers, remainder="passthrough")
                ))
            
            # Add custom steps
            if custom_steps:
                for step in custom_steps:
                    steps.append((step["name"], step["transformer"]))
            
            return Pipeline(steps)
            
        except Exception as e:
            logger.error(f"Error building pipeline: {str(e)}")
            raise
    
    def _get_llm_recommendations(
        self,
        data: pd.DataFrame,
        eda_results: Dict[str, Any]
    ) -> PreprocessingRecommendations:
        """Get LLM-powered preprocessing recommendations."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data scientist designing a preprocessing pipeline.
            Based on the EDA results, recommend appropriate preprocessing steps.
            Consider:
            1. Feature scaling needs
            2. Encoding categorical variables
            3. Handling missing values
            4. Feature selection
            
            Available transformers:
            - standard_scaler, minmax_scaler, robust_scaler
            - onehot_encoder, label_encoder, ordinal_encoder
            - simple_imputer, knn_imputer
            - drop_constant, drop_duplicate, smart_correlation"""),
            ("user", """
            Dataset Info:
            {data_info}
            
            EDA Results:
            {eda_results}
            
            Please provide preprocessing recommendations following this format:
            {format_instructions}
            """)
        ])
        
        # Prepare data info
        data_info = {
            "shape": data.shape,
            "dtypes": data.dtypes.astype(str).to_dict(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        # Get recommendations
        message = prompt.format_messages(
            data_info=data_info,
            eda_results=eda_results,
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        response = self.llm.invoke(message)
        return self.output_parser.parse(response.content)
    
    def _validate_steps(
        self,
        steps: List[PreprocessingStep],
        data: pd.DataFrame
    ) -> List[PreprocessingStep]:
        """Validate and adjust preprocessing steps if needed."""
        validated_steps = []
        
        for step in steps:
            # Validate transformer exists
            if step.transformer not in self.transformers:
                logger.warning(f"Unknown transformer: {step.transformer}, skipping")
                continue
            
            # Validate columns exist
            valid_columns = [col for col in step.columns if col in data.columns]
            if len(valid_columns) != len(step.columns):
                logger.warning(
                    f"Some columns not found for {step.name}, "
                    f"using only: {valid_columns}"
                )
                step.columns = valid_columns
            
            # Validate parameters
            transformer_class = self.transformers[step.transformer]
            try:
                # Try initializing the transformer
                transformer_class(**step.params)
                validated_steps.append(step)
            except Exception as e:
                logger.warning(
                    f"Invalid parameters for {step.name}: {str(e)}, "
                    "using default parameters"
                )
                step.params = {}
                validated_steps.append(step)
        
        return validated_steps 