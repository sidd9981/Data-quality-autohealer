"""
Ensemble Quality Classifier
Combines all individual detectors to identify quality issues
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import pickle
import sys
sys.path.append('.')

from src.detectors.schema_drift_detector import SchemaDriftDetectorTrainer
from src.detectors.distribution_shift_detector import DistributionShiftDetectorTrainer
from src.detectors.outlier_detector import OutlierDetectorTrainer
from src.detectors.type_mismatch_detector import TypeMismatchDetectorTrainer
from src.detectors.missing_data_rule_based import MissingDataRuleBasedDetector
from src.profilers.feature_engineering import QualityFeatureEngineer


class QualityEnsembleClassifier:
    """Ensemble of all quality detectors"""
    
    def __init__(self):
        self.detectors = {}
        self.feature_engineers = {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.issue_types = [
            'schema_drift',
            'distribution_shift',
            'missing_data',
            'outlier',  # Changed from 'outliers'
            'type_mismatch'
        ]
    
    def load_detectors(self, models_dir: str = 'models/detectors'):
        """Load all trained detectors"""
        
        print("Loading detectors...")
        
        # Load feature engineers
        for issue_type in self.issue_types:
            if issue_type == 'missing_data':
                continue  # Rule-based, no feature engineer needed
            
            fe_path = f'{models_dir}/{issue_type}_feature_engineer.pkl'
            try:
                fe = QualityFeatureEngineer()
                fe.load(fe_path)
                self.feature_engineers[issue_type] = fe
                print(f"  Loaded feature engineer for {issue_type}")
            except FileNotFoundError:
                print(f"  WARNING: Feature engineer not found for {issue_type}")
        
        # Load ML models
        ml_detectors = {
            'schema_drift': SchemaDriftDetectorTrainer,
            'distribution_shift': DistributionShiftDetectorTrainer,
            'outlier': OutlierDetectorTrainer,  # Changed from 'outliers'
            'type_mismatch': TypeMismatchDetectorTrainer
        }
        
        for issue_type, detector_class in ml_detectors.items():
            model_path = f'{models_dir}/{issue_type}_best.pth'
            fe = self.feature_engineers.get(issue_type)
            
            if fe is None:
                print(f"  Skipping {issue_type} - no feature engineer")
                continue
            
            try:
                input_dim = len(fe.feature_names)
                detector = detector_class(input_dim=input_dim, hidden_dims=[64, 32])
                detector.load_model(model_path)
                self.detectors[issue_type] = detector
                print(f"  Loaded {issue_type} detector")
            except FileNotFoundError:
                print(f"  WARNING: Model not found for {issue_type}")
        
        # Load rule-based missing data detector
        self.detectors['missing_data'] = MissingDataRuleBasedDetector(
            overall_threshold=0.05,
            column_threshold=0.15,
            high_missing_threshold=0.20
        )
        print(f"  Loaded missing_data detector (rule-based)")
        
        print(f"\nLoaded {len(self.detectors)} detectors")
    
    def predict_issue_types_multi(
        self,
        profile: Dict,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Predict ALL quality issues present (multi-label)
        
        Returns:
            (detected_issues, all_scores)
            detected_issues: list of issue types detected
            all_scores: dict of all detector scores
        """
        
        all_scores = {}
        
        # Get predictions from each detector
        for issue_type in self.issue_types:
            detector = self.detectors.get(issue_type)
            
            if detector is None:
                all_scores[issue_type] = 0.0
                continue
            
            if issue_type == 'missing_data':
                # Rule-based detector
                pred, prob = detector.predict(profile)
                all_scores[issue_type] = prob if pred == 1 else 0.0
            else:
                # ML detectors
                fe = self.feature_engineers.get(issue_type)
                if fe is None:
                    all_scores[issue_type] = 0.0
                    continue
                
                features = fe.transform(profile)
                _, probs = detector.predict(features.reshape(1, -1))
                all_scores[issue_type] = probs[0]
        
        # Collect all issues above threshold
        detected_issues = [
            issue for issue, score in all_scores.items()
            if score >= confidence_threshold
        ]
        
        if not detected_issues:
            detected_issues = ['clean']
        
        return detected_issues, all_scores
    
    def predict_batch(
        self,
        profiles: List[Dict],
        confidence_threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict]]:
        """Predict for multiple profiles"""
        
        results = []
        for profile in profiles:
            result = self.predict_issue_type(profile, confidence_threshold)
            results.append(result)
        
        return results


if __name__ == "__main__":
    from src.data.bad_data_generator import BadDataGenerator
    from src.profilers.spark_profiler import SparkDataProfiler
    import os
    
    print("Testing Ensemble Classifier...")
    
    # Check if models exist
    required_files = [
        'models/detectors/schema_drift_best.pth',
        'models/detectors/distribution_shift_best.pth',
        'models/detectors/outlier_best.pth',
        'models/detectors/type_mismatch_best.pth'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Missing model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease train all detectors first.")
        sys.exit(1)
    
    # Load ensemble
    ensemble = QualityEnsembleClassifier()
    ensemble.load_detectors()
    
    # Test on different issue types
    print("\n" + "="*50)
    print("Testing ensemble on different issues...")
    print("="*50)
    
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    
    test_cases = [
        ('schema_drift', {'drift_type': 'add_column'}),
        ('distribution_shift', {'column': 'salary', 'shift_factor': 2.0}),
        ('missing_data', {'missing_rate': 0.3, 'pattern': 'random'}),
        ('outlier', {'column': 'salary', 'outlier_rate': 0.1}),  # Changed from 'outliers'
        ('type_mismatch', {'error_rate': 0.2})
    ]
    
    for issue_type, kwargs in test_cases:
        print(f"\nTesting {issue_type}...")
        
        clean_df, corrupted_df = generator.generate_quality_issue_dataset(
            n_rows=200,
            issue_type=issue_type,
            **kwargs
        )
        
        profile = profiler.profile_dataset(corrupted_df, baseline_df=clean_df)
        
        detected_issues, all_scores = ensemble.predict_issue_types_multi(profile)
        
        print(f"  Detected: {detected_issues}")
        print(f"  Scores: {', '.join([f'{k}={v:.2f}' for k, v in all_scores.items()])}")
        
        if issue_type in detected_issues:
            print(f"  ✓ CORRECT")
        else:
            print(f"  ✗ MISSED")
    
    # Test clean data
    print(f"\nTesting clean data...")
    clean_df = generator.generate_clean_dataset(n_rows=200)
    profile = profiler.profile_dataset(clean_df, baseline_df=clean_df)
    
    detected_issues, all_scores = ensemble.predict_issue_types_multi(profile)
    
    print(f"  Detected: {detected_issues}")
    print(f"  Scores: {', '.join([f'{k}={v:.2f}' for k, v in all_scores.items()])}")
    
    profiler.spark.stop()
    print("\nEnsemble test complete!")