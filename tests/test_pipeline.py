import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from sklearn.datasets import make_classification, make_regression
from auto_pipeline.core import DataPipeline
from auto_pipeline.agents import EDAAgent, PreprocessingAgent

class TestDataPipeline(unittest.TestCase):
    """Test cases for the main DataPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
        
        # Create test data
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        # Create DataFrame with various data types
        self.test_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.test_data['target'] = y
        self.test_data['category'] = pd.cut(self.test_data['feature_0'], bins=3, labels=['low', 'medium', 'high'])
        self.test_data.loc[np.random.choice(self.test_data.index, 10), 'feature_1'] = np.nan
    
    def test_data_loading(self):
        """Test data loading functionality."""
        # Test DataFrame loading
        self.pipeline.load_data(self.test_data)
        self.assertIsNotNone(self.pipeline.data)
        self.assertEqual(self.pipeline.data.shape, self.test_data.shape)
        
        # Test CSV loading
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            self.test_data.to_csv(tmp.name, index=False)
            pipeline = DataPipeline()
            pipeline.load_data(tmp.name)
            self.assertEqual(pipeline.data.shape, self.test_data.shape)
    
    def test_eda_analysis(self):
        """Test EDA functionality."""
        self.pipeline.load_data(self.test_data)
        eda_results = self.pipeline.analyze()
        
        # Check basic EDA results
        self.assertIn('data_summary', eda_results)
        self.assertIn('column_analysis', eda_results)
        self.assertIn('correlations', eda_results)
        
        # Check data summary
        summary = eda_results['data_summary']
        self.assertEqual(summary['shape'], self.test_data.shape)
        self.assertGreater(summary['total_missing'], 0)
    
    def test_preprocessing_recommendations(self):
        """Test preprocessing recommendations."""
        self.pipeline.load_data(self.test_data)
        self.pipeline.analyze()
        recommendations = self.pipeline.get_recommendations()
        
        # Check recommendation structure
        self.assertIn('steps', recommendations)
        self.assertIn('reasoning', recommendations)
        
        # Check steps
        steps = recommendations['steps']
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
        
        # Verify step structure
        step = steps[0]
        self.assertIn('name', step)
        self.assertIn('transformer', step)
        self.assertIn('columns', step)
        self.assertIn('params', step)
    
    def test_pipeline_generation(self):
        """Test pipeline generation and transformation."""
        self.pipeline.load_data(self.test_data)
        self.pipeline.analyze()
        self.pipeline.get_recommendations()
        sklearn_pipeline = self.pipeline.generate()
        
        # Test pipeline transformation
        X = self.test_data.drop('target', axis=1)
        X_transformed = sklearn_pipeline.transform(X)
        
        self.assertIsNotNone(X_transformed)
        self.assertEqual(len(X_transformed), len(X))
    
    def test_pipeline_saving_loading(self):
        """Test pipeline saving and loading."""
        self.pipeline.load_data(self.test_data)
        self.pipeline.analyze()
        self.pipeline.get_recommendations()
        original_pipeline = self.pipeline.generate()
        
        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
            self.pipeline.save(tmp.name)
            
            # Load in new pipeline
            new_pipeline = DataPipeline()
            new_pipeline.load_pipeline(tmp.name)
            
            # Compare transformations
            X = self.test_data.drop('target', axis=1)
            original_transform = original_pipeline.transform(X)
            new_transform = new_pipeline.pipeline.transform(X)
            
            np.testing.assert_array_almost_equal(original_transform, new_transform)

class TestEDAAgent(unittest.TestCase):
    """Test cases for the EDA Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EDAAgent()
        
        # Create test data
        X, y = make_regression(
            n_samples=50,
            n_features=3,
            random_state=42
        )
        self.test_data = pd.DataFrame(X, columns=['num1', 'num2', 'num3'])
        self.test_data['category'] = 'A'
        self.test_data.loc[:25, 'category'] = 'B'
    
    def test_data_summary(self):
        """Test data summary generation."""
        summary = self.agent._get_data_summary(self.test_data)
        
        self.assertEqual(summary['shape'], self.test_data.shape)
        self.assertEqual(summary['duplicates'], 0)
        self.assertEqual(summary['total_missing'], 0)
    
    def test_column_analysis(self):
        """Test column-level analysis."""
        analysis = self.agent._analyze_columns(self.test_data)
        
        # Check numeric column analysis
        num_analysis = analysis['num1']
        self.assertIn('mean', num_analysis)
        self.assertIn('std', num_analysis)
        self.assertIn('skew', num_analysis)
        
        # Check categorical column analysis
        cat_analysis = analysis['category']
        self.assertIn('unique_values', cat_analysis)
        self.assertIn('most_common', cat_analysis)
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        correlations = self.agent._analyze_correlations(self.test_data)
        
        # Check numeric correlations
        self.assertIn('num1', correlations)
        self.assertIsInstance(correlations['num1'], list)
        self.assertGreater(len(correlations['num1']), 0)

class TestPreprocessingAgent(unittest.TestCase):
    """Test cases for the Preprocessing Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PreprocessingAgent()
        
        # Create test data with various preprocessing needs
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=2,
            n_redundant=1,
            random_state=42
        )
        self.test_data = pd.DataFrame(X, columns=['num1', 'num2', 'num3', 'num4'])
        self.test_data['category'] = 'A'
        self.test_data.loc[:50, 'category'] = 'B'
        self.test_data.loc[np.random.choice(self.test_data.index, 10), 'num1'] = np.nan
        
        # Create sample EDA results
        self.eda_results = {
            'data_summary': {
                'shape': self.test_data.shape,
                'total_missing': 10
            },
            'column_analysis': {
                'num1': {'missing': 10, 'dtype': 'float64'},
                'category': {'unique_values': 2, 'dtype': 'object'}
            },
            'correlations': {
                'num1': [{'feature': 'num2', 'correlation': 0.8}]
            }
        }
    
    def test_recommendation_generation(self):
        """Test preprocessing recommendation generation."""
        recommendations = self.agent.generate_recommendations(
            self.test_data,
            self.eda_results
        )
        
        self.assertIn('steps', recommendations)
        self.assertIn('reasoning', recommendations)
        
        # Check that we have recommendations for missing values
        has_imputer = any(
            step['transformer'] in ['simple_imputer', 'knn_imputer']
            for step in recommendations['steps']
        )
        self.assertTrue(has_imputer)
        
        # Check that we have recommendations for categorical encoding
        has_encoder = any(
            step['transformer'] in ['onehot_encoder', 'label_encoder', 'ordinal_encoder']
            for step in recommendations['steps']
        )
        self.assertTrue(has_encoder)
    
    def test_pipeline_building(self):
        """Test pipeline construction from recommendations."""
        recommendations = {
            'steps': [
                {
                    'name': 'imputer',
                    'transformer': 'simple_imputer',
                    'columns': ['num1'],
                    'params': {'strategy': 'mean'}
                },
                {
                    'name': 'scaler',
                    'transformer': 'standard_scaler',
                    'columns': ['num1', 'num2', 'num3', 'num4'],
                    'params': {}
                },
                {
                    'name': 'encoder',
                    'transformer': 'onehot_encoder',
                    'columns': ['category'],
                    'params': {'sparse': False}
                }
            ]
        }
        
        pipeline = self.agent.build_pipeline(recommendations)
        
        # Test pipeline transformation
        X_transformed = pipeline.transform(self.test_data)
        self.assertIsNotNone(X_transformed)
        self.assertEqual(len(X_transformed), len(self.test_data))

if __name__ == '__main__':
    unittest.main() 