import gradio as gr
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import tempfile
from ..core.data_pipeline import DataPipeline

class PipelineInterface:
    """Gradio interface for the Autonomous Data Pipeline Builder."""
    
    def __init__(self):
        """Initialize the interface and pipeline."""
        self.pipeline = DataPipeline()
        self.current_data = None
        self.current_recommendations = None
    
    def load_data(self, file: tempfile._TemporaryFileWrapper) -> Tuple[str, str]:
        """
        Load data from uploaded file.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Tuple of (status message, data preview)
        """
        try:
            # Read the data
            self.current_data = pd.read_csv(file.name)
            
            # Generate preview
            preview = self.current_data.head().to_string()
            
            return "Data loaded successfully!", preview
        except Exception as e:
            return f"Error loading data: {str(e)}", ""
    
    def analyze_data(self) -> Tuple[str, str, str]:
        """
        Run EDA on the loaded data.
        
        Returns:
            Tuple of (status message, EDA summary, visualization HTML)
        """
        if self.current_data is None:
            return "No data loaded!", "", ""
        
        try:
            # Run analysis
            eda_results = self.pipeline.analyze()
            
            # Format results
            summary = json.dumps(eda_results["data_summary"], indent=2)
            
            # Get path to generated reports
            reports_path = Path("reports")
            sweetviz_path = reports_path / "sweetviz_report.html"
            
            return (
                "Analysis completed successfully!",
                summary,
                str(sweetviz_path)
            )
        except Exception as e:
            return f"Error during analysis: {str(e)}", "", ""
    
    def get_preprocessing_recommendations(self) -> Tuple[str, str]:
        """
        Get preprocessing recommendations.
        
        Returns:
            Tuple of (status message, recommendations)
        """
        try:
            self.current_recommendations = self.pipeline.get_recommendations()
            
            # Format recommendations
            recommendations = json.dumps(
                self.current_recommendations,
                indent=2
            )
            
            return "Generated preprocessing recommendations!", recommendations
        except Exception as e:
            return f"Error getting recommendations: {str(e)}", ""
    
    def generate_pipeline(self) -> str:
        """
        Generate the preprocessing pipeline.
        
        Returns:
            Status message
        """
        try:
            self.pipeline.generate()
            return "Pipeline generated successfully!"
        except Exception as e:
            return f"Error generating pipeline: {str(e)}"
    
    def save_pipeline(self, path: str) -> str:
        """
        Save the pipeline to disk.
        
        Args:
            path: Path to save the pipeline
            
        Returns:
            Status message
        """
        try:
            self.pipeline.save(path)
            return f"Pipeline saved to {path}!"
        except Exception as e:
            return f"Error saving pipeline: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Autonomous Data Pipeline Builder") as interface:
            gr.Markdown("# 🤖 Autonomous Data Pipeline Builder")
            gr.Markdown(
                "Upload your data, and let AI help you build "
                "the perfect preprocessing pipeline!"
            )
            
            with gr.Row():
                # Data loading section
                with gr.Column():
                    file_input = gr.File(label="Upload Dataset (CSV)")
                    load_btn = gr.Button("Load Data")
                    load_status = gr.Textbox(
                        label="Load Status",
                        interactive=False
                    )
                    data_preview = gr.Textbox(
                        label="Data Preview",
                        interactive=False,
                        max_lines=10
                    )
                
                # Analysis section
                with gr.Column():
                    analyze_btn = gr.Button("Run Analysis")
                    analysis_status = gr.Textbox(
                        label="Analysis Status",
                        interactive=False
                    )
                    eda_summary = gr.JSON(
                        label="EDA Summary",
                        interactive=False
                    )
                    report_html = gr.HTML(label="EDA Report")
            
            with gr.Row():
                # Recommendations section
                with gr.Column():
                    recommend_btn = gr.Button("Get Recommendations")
                    recommend_status = gr.Textbox(
                        label="Recommendation Status",
                        interactive=False
                    )
                    recommendations = gr.JSON(
                        label="Preprocessing Recommendations",
                        interactive=False
                    )
                
                # Pipeline generation section
                with gr.Column():
                    generate_btn = gr.Button("Generate Pipeline")
                    generate_status = gr.Textbox(
                        label="Generation Status",
                        interactive=False
                    )
                    save_path = gr.Textbox(
                        label="Save Path",
                        placeholder="pipeline.joblib"
                    )
                    save_btn = gr.Button("Save Pipeline")
                    save_status = gr.Textbox(
                        label="Save Status",
                        interactive=False
                    )
            
            # Connect components
            load_btn.click(
                self.load_data,
                inputs=[file_input],
                outputs=[load_status, data_preview]
            )
            
            analyze_btn.click(
                self.analyze_data,
                outputs=[analysis_status, eda_summary, report_html]
            )
            
            recommend_btn.click(
                self.get_preprocessing_recommendations,
                outputs=[recommend_status, recommendations]
            )
            
            generate_btn.click(
                self.generate_pipeline,
                outputs=[generate_status]
            )
            
            save_btn.click(
                self.save_pipeline,
                inputs=[save_path],
                outputs=[save_status]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs) 