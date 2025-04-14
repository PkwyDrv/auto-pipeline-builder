from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import sweetviz as sv
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class EDAResults(BaseModel):
    """Pydantic model for structured EDA results."""
    data_summary: Dict[str, Any] = Field(description="Basic dataset information")
    column_analysis: Dict[str, Dict[str, Any]] = Field(description="Per-column analysis")
    correlations: Dict[str, List[Dict[str, float]]] = Field(description="Feature correlations")
    quality_issues: List[Dict[str, Any]] = Field(description="Data quality issues found")
    recommendations: List[str] = Field(description="Analysis-based recommendations")

class EDAAgent:
    """
    Agent for performing autonomous exploratory data analysis.
    Uses both traditional statistical methods and LLM-powered insights.
    """
    
    AVAILABLE_MODELS = {
        "gemini-2.0-flash": "gemini-2.0-flash",  # Free tier model
        "gemini-pro": "gemini-pro",
        "gemini-pro-vision": "gemini-pro-vision",
        "gemini-ultra": "gemini-ultra"  # Limited access
    }
    
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
        """
        Initialize the EDA agent.
        
        Args:
            model_name: Name of the LLM model to use (default: gemini-2.0-flash)
            temperature: Temperature for LLM responses
        """
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model {model_name} not found. Defaulting to gemini-2.0-flash")
            model_name = "gemini-2.0-flash"
            
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        self.output_parser = PydanticOutputParser(pydantic_object=EDAResults)
        self.reports_path = Path("reports")
        self.reports_path.mkdir(exist_ok=True)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the dataset.
        
        Args:
            data: Input DataFrame to analyze
            
        Returns:
            Dictionary containing structured EDA results
        """
        try:
            # Basic data summary
            data_summary = self._get_data_summary(data)
            
            # Column-level analysis
            column_analysis = self._analyze_columns(data)
            
            # Correlation analysis
            correlations = self._analyze_correlations(data)
            
            # Generate reports
            self._generate_reports(data)
            
            # LLM-powered analysis
            llm_insights = self._get_llm_insights(
                data_summary,
                column_analysis,
                correlations
            )
            
            # Combine all results
            results = {
                "data_summary": data_summary,
                "column_analysis": column_analysis,
                "correlations": correlations,
                "quality_issues": llm_insights.quality_issues,
                "recommendations": llm_insights.recommendations
            }
            
            logger.info("Completed EDA analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error in EDA analysis: {str(e)}")
            raise
    
    def _get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset summary statistics."""
        return {
            "shape": list(data.shape),
            "memory_usage": int(data.memory_usage(deep=True).sum()),
            "duplicates": int(data.duplicated().sum()),
            "total_missing": int(data.isnull().sum().sum())
        }
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Perform detailed analysis of each column."""
        analysis = {}
        
        for column in data.columns:
            col_data = data[column]
            
            # Basic stats that work for all types
            stats = {
                "dtype": str(col_data.dtype),
                "missing": int(col_data.isnull().sum()),
                "missing_pct": float((col_data.isnull().sum() / len(col_data)) * 100)
            }
            
            # Handle different data types
            if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_categorical_dtype(col_data):
                numeric_stats = {
                    "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                    "std": float(col_data.std()) if not col_data.isnull().all() else None,
                    "min": float(col_data.min()) if not col_data.isnull().all() else None,
                    "max": float(col_data.max()) if not col_data.isnull().all() else None,
                    "skew": float(col_data.skew()) if not col_data.isnull().all() else None,
                    "kurtosis": float(col_data.kurtosis()) if not col_data.isnull().all() else None
                }
                stats.update(numeric_stats)
            elif pd.api.types.is_categorical_dtype(col_data):
                cat_stats = {
                    "unique_values": int(col_data.nunique()),
                    "categories": list(col_data.cat.categories),
                    "ordered": bool(col_data.cat.ordered),
                    "value_counts": col_data.value_counts().head().to_dict()
                }
                stats.update(cat_stats)
            else:
                # Handle string/object columns
                stats.update({
                    "unique_values": int(col_data.nunique()),
                    "most_common": dict(col_data.value_counts().head().to_dict())
                })
            
            analysis[column] = stats
        
        return analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
        """Analyze correlations between features."""
        # Get only numeric columns, excluding categorical
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        if not numeric_data.empty and len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            
            correlations = {}
            for column in corr_matrix.columns:
                # Get top 5 correlations for each feature
                top_corr = (corr_matrix[column]
                           .sort_values(ascending=False)
                           .head(6)  # 6 to include self-correlation
                           .drop(column)  # remove self-correlation
                           .to_dict())
                correlations[column] = [
                    {"feature": k, "correlation": float(v)}
                    for k, v in top_corr.items()
                ]
        else:
            correlations = {}
        
        return correlations
    
    def _generate_reports(self, data: pd.DataFrame) -> None:
        """Generate and save detailed EDA reports."""
        try:
            # Sweetviz report
            sweet_report = sv.analyze(data)
            sweet_report.show_html(
                str(self.reports_path / "sweetviz_report.html")
            )
            
            # Pandas profiling report
            profile = ProfileReport(
                data,
                title="Pandas Profiling Report",
                explorative=True
            )
            profile.to_file(str(self.reports_path / "profiling_report.html"))
            
            logger.info("Generated EDA reports")
            
        except Exception as e:
            logger.warning(f"Error generating reports: {str(e)}")
    
    def _get_llm_insights(
        self,
        data_summary: Dict[str, Any],
        column_analysis: Dict[str, Dict[str, Any]],
        correlations: Dict[str, List[Dict[str, float]]]
    ) -> EDAResults:
        """Get LLM-powered insights from the analysis results."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data scientist analyzing a dataset.
            Based on the provided analysis results, identify key insights,
            data quality issues, and make recommendations for preprocessing
            and modeling."""),
            ("user", """
            Data Summary: {data_summary}
            Column Analysis: {column_analysis}
            Correlations: {correlations}
            
            Please provide structured insights following this format:
            {format_instructions}
            """)
        ])
        
        # Prepare the message
        message = prompt.format_messages(
            data_summary=json.dumps(data_summary, indent=2),
            column_analysis=json.dumps(column_analysis, indent=2),
            correlations=json.dumps(correlations, indent=2),
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        # Get LLM response
        response = self.llm.invoke(message)
        
        # Parse the response
        return self.output_parser.parse(response.content) 