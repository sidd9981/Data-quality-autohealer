"""
Rule-based Missing Data Detector
Sometimes simple rules work better than ML
"""

from typing import Dict, Tuple
import numpy as np


class MissingDataRuleBasedDetector:
    """Rule-based detector for missing data issues"""
    
    def __init__(
        self,
        overall_threshold: float = 0.05,  # Lower threshold (was 0.10)
        column_threshold: float = 0.15,   # Lower threshold (was 0.30)
        high_missing_threshold: float = 0.20  # Lower threshold (was 0.50)
    ):
        self.overall_threshold = overall_threshold
        self.column_threshold = column_threshold
        self.high_missing_threshold = high_missing_threshold
    
    def predict(self, profile: Dict) -> Tuple[int, float]:
        """
        Predict if dataset has missing data issues
        
        Returns:
            (prediction, confidence) where prediction is 0 or 1
        """
        
        overall_rate = profile.get('overall_missing_rate', 0.0)
        max_rate = profile.get('max_missing_rate', 0.0)
        columns_with_missing = profile.get('columns_with_missing', 0)
        high_missing_count = profile.get('high_missing_columns_count', 0)
        
        # Rule 1: Overall missing rate too high
        if overall_rate > self.overall_threshold:
            confidence = min(overall_rate / self.overall_threshold, 1.0)
            return 1, confidence
        
        # Rule 2: Any column has very high missing rate
        if max_rate > self.high_missing_threshold:
            confidence = min(max_rate / self.high_missing_threshold, 1.0)
            return 1, confidence
        
        # Rule 3: Multiple columns with even small missing rates
        if columns_with_missing >= 2 and overall_rate > 0.03:
            confidence = min(overall_rate / 0.03, 1.0)
            return 1, confidence
        
        # No issue detected
        confidence = 1.0 - overall_rate / self.overall_threshold
        return 0, max(0.0, confidence)
    
    def predict_batch(self, profiles: list) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for multiple profiles"""
        
        preds = []
        probs = []
        
        for profile in profiles:
            pred, prob = self.predict(profile)
            preds.append(pred)
            probs.append(prob)
        
        return np.array(preds), np.array(probs)


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.data.bad_data_generator import BadDataGenerator
    from src.profilers.spark_profiler import SparkDataProfiler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("Testing Rule-Based Missing Data Detector...")
    
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    detector = MissingDataRuleBasedDetector()
    
    # Generate test samples
    print("\nGenerating test samples...")
    profiles = []
    labels = []
    
    # Missing data samples
    for i in range(50):
        clean_df, missing_df = generator.generate_quality_issue_dataset(
            n_rows=200,
            issue_type='missing_data',
            missing_rate=np.random.uniform(0.1, 0.5),
            pattern=np.random.choice(['random', 'systematic', 'streak'])
        )
        profile = profiler.profile_dataset(missing_df, baseline_df=clean_df)
        profiles.append(profile)
        labels.append(1)
    
    # Clean samples
    for i in range(50):
        clean_df = generator.generate_clean_dataset(n_rows=200)
        profile = profiler.profile_dataset(clean_df, baseline_df=clean_df)
        profiles.append(profile)
        labels.append(0)
    
    print(f"Generated {len(profiles)} test samples")
    
    # Predict
    print("\nTesting detector...")
    preds, probs = detector.predict_batch(profiles)
    labels = np.array(labels)
    
    # Evaluate
    print("\nResults:")
    print(f"  Accuracy: {accuracy_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1: {f1_score(labels, preds, zero_division=0):.4f}")
    
    # Show some examples
    print("\nSample predictions:")
    for i in range(5):
        print(f"  Sample {i}: True={labels[i]}, Pred={preds[i]}, Conf={probs[i]:.4f}")
    
    profiler.spark.stop()
    print("\nRule-based detector test complete!")