"""
Autonomous Data Pipeline Builder Demo

This script demonstrates how to use the Autonomous Data Pipeline Builder to:
1. Load and analyze a dataset
2. Get AI-powered preprocessing recommendations
3. Generate and save a reusable scikit-learn pipeline

We'll show both the Python API and the Gradio interface.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from auto_pipeline.core import DataPipeline
from auto_pipeline.utils.interface import PipelineInterface

def create_sample_dataset():
    """Create a sample dataset with various features."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Add categorical column
    df['category'] = pd.cut(df['feature_0'], bins=3, labels=['low', 'medium', 'high'])

    # Add some missing values
    df.loc[np.random.choice(df.index, 100), 'feature_1'] = np.nan

    # Save to CSV
    df.to_csv('sample_data.csv', index=False)
    return df

def demo_python_api(data):
    """Demonstrate using the pipeline builder through Python API."""
    print("\n=== Using Python API ===\n")
    
    # Initialize the pipeline
    pipeline = DataPipeline()

    # Load the data
    print("Loading data...")
    pipeline.load_data(data)

    # Run EDA
    print("\nRunning EDA...")
    eda_results = pipeline.analyze()

    # Display data summary
    print("\nDataset Summary:")
    print("-------------")
    for key, value in eda_results['data_summary'].items():
        print(f"{key}: {value}")

    # Get preprocessing recommendations
    print("\nGetting preprocessing recommendations...")
    recommendations = pipeline.get_recommendations()

    print("\nPreprocessing Recommendations:")
    print("-------------------------")
    for i, reason in enumerate(recommendations['reasoning'], 1):
        print(f"{i}. {reason}")

    # Generate the pipeline
    print("\nGenerating pipeline...")
    sklearn_pipeline = pipeline.generate()

    # Save it
    pipeline.save('my_pipeline.joblib')
    print("Pipeline saved to 'my_pipeline.joblib'")

    return sklearn_pipeline

def demo_gradio_interface():
    """Demonstrate using the pipeline builder through Gradio interface."""
    print("\n=== Using Gradio Interface ===\n")
    print("Launching Gradio interface...")
    
    interface = PipelineInterface()
    interface.launch(share=True)

def demo_using_pipeline(data, pipeline):
    """Demonstrate using the generated pipeline."""
    print("\n=== Using Generated Pipeline ===\n")
    
    # Transform new data
    X_transformed = pipeline.transform(data.drop('target', axis=1))
    print("Transformed data shape:", X_transformed.shape)
    print("\nFirst few rows of transformed data:")
    print(pd.DataFrame(X_transformed[:5]))

def main():
    """Run the complete demo."""
    print("Creating sample dataset...")
    data = create_sample_dataset()
    
    # Demo Python API
    pipeline = demo_python_api(data)
    
    # Demo using the pipeline
    demo_using_pipeline(data, pipeline)
    
    # Demo Gradio interface
    demo_gradio_interface()

if __name__ == "__main__":
    main() 