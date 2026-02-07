"""
Feature engineering for quality metrics
Convert profiling results into ML-ready features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
import pickle


class QualityFeatureEngineer:
    """Convert quality profiles into ML features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fitted = False
    
    def extract_numerical_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Extract only numerical features from profile"""
        
        numerical_features = {}
        
        for key, value in profile.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numerical_features[key] = float(value)
            elif isinstance(value, bool):
                numerical_features[key] = float(value)
        
        return numerical_features
    
    def create_schema_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Create schema-specific features"""
        
        features = {}
        
        # Basic counts
        features['column_count'] = profile.get('column_count', 0)
        features['nullable_columns'] = profile.get('nullable_columns', 0)
        features['new_columns_count'] = profile.get('new_columns_count', 0)
        features['removed_columns_count'] = profile.get('removed_columns_count', 0)
        features['type_changes_count'] = profile.get('type_changes_count', 0)
        
        # Ratios
        if features['column_count'] > 0:
            features['nullable_ratio'] = features['nullable_columns'] / features['column_count']
            features['new_columns_ratio'] = features['new_columns_count'] / features['column_count']
            features['removed_columns_ratio'] = features['removed_columns_count'] / features['column_count']
        else:
            features['nullable_ratio'] = 0.0
            features['new_columns_ratio'] = 0.0
            features['removed_columns_ratio'] = 0.0
        
        # Schema change magnitude
        features['schema_change_magnitude'] = (
            features['new_columns_count'] + 
            features['removed_columns_count'] + 
            features['type_changes_count']
        )
        
        return features
    
    def create_distribution_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Create distribution shift features"""
        
        features = {}
        
        # Collect all mean_shift and variance_ratio features
        mean_shifts = []
        variance_ratios = []
        
        for key, value in profile.items():
            if '_mean_shift' in key and isinstance(value, (int, float)):
                mean_shifts.append(float(value))
            elif '_variance_ratio' in key and isinstance(value, (int, float)):
                variance_ratios.append(float(value))
        
        if mean_shifts:
            features['max_mean_shift'] = max(mean_shifts)
            features['avg_mean_shift'] = np.mean(mean_shifts)
            features['std_mean_shift'] = np.std(mean_shifts)
        else:
            features['max_mean_shift'] = 0.0
            features['avg_mean_shift'] = 0.0
            features['std_mean_shift'] = 0.0
        
        if variance_ratios:
            features['max_variance_ratio'] = max(variance_ratios)
            features['avg_variance_ratio'] = np.mean(variance_ratios)
            features['std_variance_ratio'] = np.std(variance_ratios)
        else:
            features['max_variance_ratio'] = 1.0
            features['avg_variance_ratio'] = 1.0
            features['std_variance_ratio'] = 0.0
        
        return features
    
    def create_missing_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Create missing data features"""
        
        features = {}
        
        # Overall missing rate
        features['overall_missing_rate'] = profile.get('overall_missing_rate', 0.0)
        features['high_missing_columns_count'] = profile.get('high_missing_columns_count', 0)
        
        # Collect per-column missing rates
        missing_rates = []
        for key, value in profile.items():
            if '_missing_rate' in key and isinstance(value, (int, float)):
                missing_rates.append(float(value))
        
        if missing_rates:
            features['max_missing_rate'] = max(missing_rates)
            features['avg_missing_rate'] = np.mean(missing_rates)
            features['std_missing_rate'] = np.std(missing_rates)
            features['missing_columns_count'] = sum(1 for r in missing_rates if r > 0)
        else:
            features['max_missing_rate'] = 0.0
            features['avg_missing_rate'] = 0.0
            features['std_missing_rate'] = 0.0
            features['missing_columns_count'] = 0
        
        return features
    
    def create_outlier_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Create outlier features"""
        
        features = {}
        
        # Collect outlier rates
        outlier_rates = []
        for key, value in profile.items():
            if '_outlier_rate' in key and isinstance(value, (int, float)):
                outlier_rates.append(float(value))
        
        if outlier_rates:
            features['max_outlier_rate'] = max(outlier_rates)
            features['avg_outlier_rate'] = np.mean(outlier_rates)
            features['std_outlier_rate'] = np.std(outlier_rates)
            features['columns_with_outliers'] = sum(1 for r in outlier_rates if r > 0.01)
        else:
            features['max_outlier_rate'] = 0.0
            features['avg_outlier_rate'] = 0.0
            features['std_outlier_rate'] = 0.0
            features['columns_with_outliers'] = 0
        
        return features
    
    def create_statistical_features(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Create aggregate statistical features"""
        
        features = {}
        
        # Collect cardinality values
        cardinalities = []
        for key, value in profile.items():
            if '_cardinality' in key and isinstance(value, (int, float)):
                cardinalities.append(float(value))
        
        if cardinalities:
            features['avg_cardinality'] = np.mean(cardinalities)
            features['max_cardinality'] = max(cardinalities)
            features['low_cardinality_columns'] = sum(1 for c in cardinalities if c < 0.01)
        else:
            features['avg_cardinality'] = 0.0
            features['max_cardinality'] = 0.0
            features['low_cardinality_columns'] = 0
        
        return features
    
    def engineer_features(self, profile: Dict[str, Any]) -> np.ndarray:
        """
        Convert quality profile to feature vector
        
        Args:
            profile: Quality metrics dictionary from profiler
        
        Returns:
            Feature vector as numpy array
        """
        
        # Create feature groups
        schema_features = self.create_schema_features(profile)
        distribution_features = self.create_distribution_features(profile)
        missing_features = self.create_missing_features(profile)
        outlier_features = self.create_outlier_features(profile)
        statistical_features = self.create_statistical_features(profile)
        
        # Combine all features
        all_features = {}
        all_features.update(schema_features)
        all_features.update(distribution_features)
        all_features.update(missing_features)
        all_features.update(outlier_features)
        all_features.update(statistical_features)
        
        # Convert to consistent feature vector
        if not self.fitted:
            self.feature_names = sorted(all_features.keys())
        
        feature_vector = np.array([all_features.get(name, 0.0) for name in self.feature_names])
        
        return feature_vector
    
    def fit(self, profiles: List[Dict[str, Any]]):
        """Fit the feature engineer on training profiles"""
        
        # Extract features from all profiles
        feature_vectors = []
        for profile in profiles:
            features = self.engineer_features(profile)
            feature_vectors.append(features)
        
        X = np.array(feature_vectors)
        
        # Fit scaler
        self.scaler.fit(X)
        self.fitted = True
        
        return self
    
    def transform(self, profile: Dict[str, Any]) -> np.ndarray:
        """Transform a single profile to scaled features"""
        
        features = self.engineer_features(profile)
        
        if self.fitted:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            return features_scaled[0]
        else:
            return features
    
    def fit_transform(self, profiles: List[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform training profiles"""
        
        self.fit(profiles)
        
        transformed = []
        for profile in profiles:
            features = self.transform(profile)
            transformed.append(features)
        
        return np.array(transformed)
    
    def save(self, path: str):
        """Save feature engineer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'fitted': self.fitted
            }, f)
    
    def load(self, path: str):
        """Load feature engineer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.fitted = data['fitted']


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('.')
    from src.data.bad_data_generator import BadDataGenerator
    from src.profilers.spark_profiler import SparkDataProfiler
    
    print("Testing QualityFeatureEngineer...")
    
    # Generate test data
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    
    # Create multiple profiles
    print("\nGenerating profiles...")
    profiles = []
    
    for i in range(10):
        clean_df, corrupted_df = generator.generate_quality_issue_dataset(
            n_rows=500,
            issue_type='schema_drift'
        )
        
        profile = profiler.profile_dataset(corrupted_df, baseline_df=clean_df)
        profiles.append(profile)
    
    print(f"Generated {len(profiles)} profiles")
    
    # Engineer features
    engineer = QualityFeatureEngineer()
    
    print("\nExtracting features...")
    X = engineer.fit_transform(profiles)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(engineer.feature_names)}")
    print(f"Feature names: {engineer.feature_names[:10]}...")
    print(f"Sample feature vector: {X[0][:10]}")
    
    # Test single transform
    print("\nTesting single profile transform...")
    new_profile = profiler.profile_dataset(corrupted_df, baseline_df=clean_df)
    features = engineer.transform(new_profile)
    print(f"Transformed features shape: {features.shape}")
    
    profiler.spark.stop()
    print("\nFeature engineering test complete!")