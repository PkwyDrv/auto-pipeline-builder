from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from langchain.chat_models import ChatOpenAI
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
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        """
        Initialize the EDA agent.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM responses
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
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
            "shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum(),
            "duplicates": data.duplicated().sum(),
            "total_missing": data.isnull().sum().sum()
        }
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Perform detailed analysis of each column."""
        analysis = {}
        
        for column in data.columns:
            col_data = data[column]
            col_type = str(col_data.dtype)
            
            # Basic stats
            stats = {
                "dtype": col_type,
                "missing": col_data.isnull().sum(),
                "missing_pct": (col_data.isnull().sum() / len(col_data)) * 100
            }
            
            # Type-specific analysis
            if np.issubdtype(col_data.dtype, np.number):
                stats.update({
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "skew": col_data.skew(),
                    "kurtosis": col_data.kurtosis()
                })
            else:
                stats.update({
                    "unique_values": col_data.nunique(),
                    "most_common": col_data.value_counts().head().to_dict()
                })
            
            analysis[column] = stats
        
        return analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
        """Analyze correlations between features."""
        # Numeric correlations
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
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
                    {"feature": k, "correlation": v}
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