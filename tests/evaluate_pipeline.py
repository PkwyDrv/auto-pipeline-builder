"""
Evaluation script for the Autonomous Data Pipeline Builder.
Tests the pipeline on various real-world datasets and scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_boston, load_diabetes, load_breast_cancer,
    fetch_california_housing
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import json
from pathlib import Path
from auto_pipeline.core import DataPipeline

class PipelineEvaluator:
    """Evaluates the pipeline on various datasets and metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        self.reports_dir = Path("evaluation_reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def evaluate_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str,
        task_type: str = "regression"
    ):
        """
        Evaluate pipeline on a single dataset.
        
        Args:
            X: Feature DataFrame
            y: Target series
            dataset_name: Name of the dataset
            task_type: Type of task ("regression" or "classification")
        """
        print(f"\nEvaluating on {dataset_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        try:
            # Time the pipeline process
            start_time = time.time()
            
            # Initialize and run pipeline
            pipeline = DataPipeline()
            pipeline.load_data(pd.concat([X_train, pd.Series(y_train, name='target')], axis=1))
            
            # Run analysis
            eda_results = pipeline.analyze()
            
            # Get recommendations
            recommendations = pipeline.get_recommendations()
            
            # Generate preprocessing pipeline
            preprocessing_pipeline = pipeline.generate()
            
            # Transform data
            X_train_transformed = preprocessing_pipeline.transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)
            
            # Train and evaluate model
            if task_type == "regression":
                model = RandomForestRegressor(random_state=42)
                metric = mean_squared_error
                metric_name = "MSE"
            else:
                model = RandomForestClassifier(random_state=42)
                metric = accuracy_score
                metric_name = "Accuracy"
            
            # Fit and predict
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            
            # Calculate metrics
            score = metric(y_test, y_pred)
            
            # Calculate timing
            total_time = time.time() - start_time
            
            # Store results
            self.results[dataset_name] = {
                "success": True,
                "time_taken": total_time,
                f"{metric_name.lower()}": score,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_preprocessing_steps": len(recommendations["steps"]),
                "recommendations": recommendations
            }
            
            print(f"✓ Evaluation completed: {metric_name} = {score:.4f}")
            
        except Exception as e:
            print(f"✗ Error evaluating {dataset_name}: {str(e)}")
            self.results[dataset_name] = {
                "success": False,
                "error": str(e)
            }
    
    def run_full_evaluation(self):
        """Run evaluation on all test datasets."""
        # Regression datasets
        try:
            boston = load_boston()
            self.evaluate_dataset(
                pd.DataFrame(boston.data, columns=boston.feature_names),
                pd.Series(boston.target),
                "Boston Housing",
                "regression"
            )
        except Exception as e:
            print(f"Error loading Boston dataset: {str(e)}")
        
        try:
            diabetes = load_diabetes()
            self.evaluate_dataset(
                pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
                pd.Series(diabetes.target),
                "Diabetes",
                "regression"
            )
        except Exception as e:
            print(f"Error loading Diabetes dataset: {str(e)}")
        
        try:
            california = fetch_california_housing()
            self.evaluate_dataset(
                pd.DataFrame(california.data, columns=california.feature_names),
                pd.Series(california.target),
                "California Housing",
                "regression"
            )
        except Exception as e:
            print(f"Error loading California Housing dataset: {str(e)}")
        
        # Classification dataset
        try:
            breast_cancer = load_breast_cancer()
            self.evaluate_dataset(
                pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names),
                pd.Series(breast_cancer.target),
                "Breast Cancer",
                "classification"
            )
        except Exception as e:
            print(f"Error loading Breast Cancer dataset: {str(e)}")
    
    def save_results(self):
        """Save evaluation results to file."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.reports_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print("------------------")
        for dataset, result in self.results.items():
            if result["success"]:
                metrics = [
                    f"{k} = {v:.4f}" 
                    for k, v in result.items() 
                    if k in ["mse", "accuracy"]
                ]
                print(f"{dataset}: Success ({', '.join(metrics)})")
            else:
                print(f"{dataset}: Failed - {result['error']}")

def main():
    """Run the evaluation."""
    evaluator = PipelineEvaluator()
    evaluator.run_full_evaluation()
    evaluator.save_results()

if __name__ == "__main__":
    main() 