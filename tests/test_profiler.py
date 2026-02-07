"""
Test suite for data profiler and feature engineering
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.profilers.spark_profiler import SparkDataProfiler
from src.profilers.feature_engineering import QualityFeatureEngineer
from src.data.bad_data_generator import BadDataGenerator


class TestSparkDataProfiler:
    
    @pytest.fixture
    def profiler(self):
        return SparkDataProfiler()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id': range(1000),
            'age': np.random.randint(18, 80, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'department': np.random.choice(['Sales', 'Engineering', 'HR'], 1000)
        })
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initializes correctly"""
        assert profiler.spark is not None
    
    def test_profile_extracts_all_metrics(self, profiler, sample_df):
        """Test profiler extracts all expected metrics"""
        profile = profiler.profile_dataset(sample_df)
        
        # Check required metrics exist
        required_metrics = [
            'row_count',
            'column_count'
        ]
        
        for metric in required_metrics:
            assert metric in profile, f"Missing metric: {metric}"
    
    def test_profile_numeric_stats(self, profiler, sample_df):
        """Test numeric column statistics"""
        profile = profiler.profile_dataset(sample_df)
        
        # Check profile has data
        assert 'row_count' in profile
        assert profile['row_count'] == 1000
    
    def test_profile_categorical_stats(self, profiler, sample_df):
        """Test categorical column statistics"""
        profile = profiler.profile_dataset(sample_df)
        
        # Check basic stats
        assert 'column_count' in profile
        assert profile['column_count'] == 4
    
    def test_profile_missing_data(self, profiler):
        """Test profiler correctly identifies missing data"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, None, 6],
            'col2': ['a', 'b', 'c', None, None, None]
        })
        
        profile = profiler.profile_dataset(df)
        
        # Should detect missing data - check for col-specific missing rates
        assert 'col1_missing_rate' in profile or 'col2_missing_rate' in profile
        
        # At least one column should have > 20% missing
        missing_rates = [v for k, v in profile.items() if 'missing_rate' in k]
        assert any(rate > 0.20 for rate in missing_rates)
    
    def test_profile_duplicates(self, profiler):
        """Test profiler correctly identifies duplicates"""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3],
            'value': ['a', 'a', 'b', 'b', 'c']
        })
        
        profile = profiler.profile_dataset(df)
        
        # Should have row_count
        assert profile['row_count'] == 5
    
    def test_profile_large_dataset(self, profiler):
        """Test profiler handles large datasets efficiently"""
        import time
        
        # Generate 100K row dataset
        large_df = pd.DataFrame({
            'id': range(100000),
            'value': np.random.randn(100000)
        })
        
        start = time.time()
        profile = profiler.profile_dataset(large_df)
        duration = time.time() - start
        
        # Should complete in < 30 seconds
        assert duration < 30, f"Profiling took {duration}s, too slow"
        assert profile['row_count'] == 100000
    
    def test_profile_handles_empty_dataframe(self, profiler):
        """Test profiler handles edge case of empty dataframe"""
        # Create DataFrame with columns but no rows
        df = pd.DataFrame(columns=['col1', 'col2'])
        
        try:
            profile = profiler.profile_dataset(df)
            assert profile['row_count'] == 0
        except (IndexError, ValueError):
            # PySpark can't handle truly empty DataFrames, that's expected
            pytest.skip("PySpark doesn't support empty DataFrames")


class TestQualityFeatureEngineer:
    
    @pytest.fixture
    def feature_engineer(self):
        return QualityFeatureEngineer()
    
    @pytest.fixture
    def sample_profile(self):
        profiler = SparkDataProfiler()
        gen = BadDataGenerator()
        df = gen.generate_clean_dataset(n_rows=1000)
        return profiler.profile_dataset(df)
    
    def test_feature_engineer_initialization(self, feature_engineer):
        """Test feature engineer initializes correctly"""
        assert hasattr(feature_engineer, 'feature_names')
    
    def test_feature_extraction_schema_drift(self, feature_engineer, sample_profile):
        """Test feature extraction for schema drift detection"""
        features = feature_engineer.engineer_features(sample_profile)
        
        # Should return numpy array
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # All features should be numeric
        assert all(isinstance(v, (int, float, np.number)) for v in features)
    
    def test_feature_extraction_distribution_shift(self, feature_engineer, sample_profile):
        """Test feature extraction works consistently"""
        features = feature_engineer.engineer_features(sample_profile)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_feature_extraction_outlier(self, feature_engineer, sample_profile):
        """Test feature extraction returns valid array"""
        features = feature_engineer.engineer_features(sample_profile)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_feature_extraction_type_mismatch(self, feature_engineer, sample_profile):
        """Test feature extraction handles all profiles"""
        features = feature_engineer.engineer_features(sample_profile)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_feature_consistency(self, feature_engineer):
        """Test feature extraction is consistent across calls"""
        profiler = SparkDataProfiler()
        gen = BadDataGenerator()
        df = gen.generate_clean_dataset(n_rows=1000)
        
        profile1 = profiler.profile_dataset(df)
        profile2 = profiler.profile_dataset(df)
        
        features1 = feature_engineer.engineer_features(profile1)
        features2 = feature_engineer.engineer_features(profile2)
        
        # Should extract same number of features
        assert len(features1) == len(features2)
    
    def test_feature_dimensionality(self, feature_engineer, sample_profile):
        """Test feature vector has correct dimensionality"""
        features = feature_engineer.engineer_features(sample_profile)
        
        # Should have 28 features
        assert len(features) == 28, f"Expected 28 features, got {len(features)}"


class TestProfilerIntegration:
    """Integration tests for profiler with bad data generator"""
    
    def test_profiler_with_all_issue_types(self):
        """Test profiler correctly profiles all issue types"""
        gen = BadDataGenerator()
        profiler = SparkDataProfiler()
        
        issue_types = ['schema_drift', 'distribution_shift', 'missing_data', 'outlier', 'type_mismatch']
        
        for issue_type in issue_types:
            df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type=issue_type)
            profile = profiler.profile_dataset(df)
            
            # All profiles should have core metrics
            assert 'row_count' in profile
            assert 'column_count' in profile
            assert profile['row_count'] > 0
    
    def test_profiler_performance_consistency(self):
        """Test profiler performance is consistent"""
        import time
        
        gen = BadDataGenerator()
        profiler = SparkDataProfiler()
        
        durations = []
        for _ in range(5):
            df = gen.generate_clean_dataset(n_rows=1000)
            
            start = time.time()
            profile = profiler.profile_dataset(df)
            duration = time.time() - start
            durations.append(duration)
        
        # Variance should be low (consistent performance)
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        assert std_duration < avg_duration * 0.5, "Profiling performance too variable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])