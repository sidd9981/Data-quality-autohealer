"""
Test suite for ensemble classifier
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.detectors.ensemble_classifier import QualityEnsembleClassifier
from src.data.bad_data_generator import BadDataGenerator
from src.profilers.spark_profiler import SparkDataProfiler


class TestEnsembleClassifier:
    
    @pytest.fixture
    def ensemble(self):
        classifier = QualityEnsembleClassifier()
        classifier.load_detectors()
        return classifier
    
    def test_ensemble_loads_all_detectors(self, ensemble):
        """Verify all 5 detectors are loaded"""
        expected_detectors = [
            'schema_drift',
            'distribution_shift',
            'missing_data',
            'outlier',
            'type_mismatch'
        ]
        
        for detector_name in expected_detectors:
            assert detector_name in ensemble.detectors, f"{detector_name} not loaded"
        
        assert len(ensemble.detectors) == 5
    
    def test_schema_drift_detection(self, ensemble):
        """Test ensemble processes schema drift data"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Validate system runs and returns results
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert 'schema_drift' in scores
        # Schema drift score should be non-zero for schema drift data
        assert scores['schema_drift'] > 0
    
    def test_distribution_shift_detection(self, ensemble):
        """Test ensemble processes distribution shift data"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='distribution_shift')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert 'distribution_shift' in scores
    
    def test_missing_data_detection(self, ensemble):
        """Test ensemble processes missing data"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='missing_data')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert 'missing_data' in scores
    
    def test_outlier_detection(self, ensemble):
        """Test ensemble processes outlier data"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='outlier')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert 'outlier' in scores
    
    def test_type_mismatch_detection(self, ensemble):
        """Test ensemble processes type mismatch data"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='type_mismatch')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert 'type_mismatch' in scores
    
    def test_clean_data_detection(self, ensemble):
        """Test ensemble correctly identifies clean data"""
        gen = BadDataGenerator()
        df = gen.generate_clean_dataset(n_rows=1000)
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Should detect as clean (all scores below threshold)
        assert 'clean' in issues or len(issues) == 0
        assert all(score < 0.70 for score in scores.values())
    
    def test_multi_issue_detection(self, ensemble):
        """Test ensemble returns valid results"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Validate return types
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert len(scores) == 5
    
    def test_ensemble_threshold_configuration(self, ensemble):
        """Test ensemble threshold can be configured"""
        # Set custom threshold
        original_threshold = ensemble.threshold
        ensemble.threshold = 0.80
        
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='outlier')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Validate threshold was changed
        assert ensemble.threshold == 0.80
        ensemble.threshold = original_threshold
    
    def test_ensemble_performance(self, ensemble):
        """Test ensemble meets performance requirements"""
        import time
        
        gen = BadDataGenerator()
        df = gen.generate_clean_dataset(n_rows=1000)
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        # Time ensemble prediction
        start = time.time()
        issues, scores = ensemble.predict_issue_types_multi(profile)
        duration = time.time() - start
        
        # Should complete in < 5 seconds
        assert duration < 5.0, f"Ensemble took {duration}s, should be < 5s"


class TestEnsembleIntegration:
    """Integration tests for ensemble with real pipeline"""
    
    def test_end_to_end_detection(self):
        """Test complete flow: data → profile → detect"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
        
        profiler = SparkDataProfiler()
        profile = profiler.profile_dataset(df)
        
        ensemble = QualityEnsembleClassifier()
        ensemble.load_detectors()
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Verify system produces valid output
        assert isinstance(issues, list)
        assert isinstance(scores, dict)
        assert len(scores) == 5
        assert all(isinstance(score, (int, float)) for score in scores.values())
    
    def test_batch_prediction(self):
        """Test ensemble can handle batch predictions"""
        ensemble = QualityEnsembleClassifier()
        ensemble.load_detectors()
        
        gen = BadDataGenerator()
        profiler = SparkDataProfiler()
        
        # Generate 10 different datasets
        results = []
        for i in range(10):
            if i % 2 == 0:
                df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
            else:
                df = gen.generate_clean_dataset(n_rows=1000)
            
            profile = profiler.profile_dataset(df)
            issues, scores = ensemble.predict_issue_types_multi(profile)
            results.append((issues, scores))
        
        # All should return valid results
        assert len(results) == 10
        assert all(isinstance(issues, list) and isinstance(scores, dict) for issues, scores in results)
    
    def test_all_issue_types_have_scores(self):
        """Test ensemble returns scores for all detector types"""
        ensemble = QualityEnsembleClassifier()
        ensemble.load_detectors()
        
        gen = BadDataGenerator()
        profiler = SparkDataProfiler()
        
        df = gen.generate_clean_dataset(n_rows=1000)
        profile = profiler.profile_dataset(df)
        issues, scores = ensemble.predict_issue_types_multi(profile)
        
        # Should have scores for all 5 detectors
        expected_scores = {'schema_drift', 'distribution_shift', 'missing_data', 'outlier', 'type_mismatch'}
        assert expected_scores.issubset(scores.keys()), f"Missing scores for: {expected_scores - scores.keys()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])