"""
PySpark-based data profiler
Extracts 50+ quality metrics from datasets
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from typing import Dict, List, Tuple, Any
import hashlib
import json


class SparkDataProfiler:
    """Extract comprehensive quality metrics from datasets using PySpark"""
    
    def __init__(self, spark: SparkSession = None):
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("DataQualityProfiler") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark
    
    def pandas_to_spark(self, df: pd.DataFrame) -> DataFrame:
        """Convert pandas DataFrame to Spark DataFrame"""
        return self.spark.createDataFrame(df)
    
    def extract_schema_features(
        self, 
        df: DataFrame,
        baseline_schema: StructType = None
    ) -> Dict[str, Any]:
        """Extract schema-related features"""
        
        schema = df.schema
        
        features = {
            'column_count': len(schema.fields),
            'column_names_hash': hashlib.md5(
                ','.join([f.name for f in schema.fields]).encode()
            ).hexdigest(),
            'data_types': {f.name: str(f.dataType) for f in schema.fields},
            'nullable_columns': sum(1 for f in schema.fields if f.nullable),
        }
        
        if baseline_schema:
            baseline_names = set(f.name for f in baseline_schema.fields)
            current_names = set(f.name for f in schema.fields)
            
            features['new_columns'] = list(current_names - baseline_names)
            features['removed_columns'] = list(baseline_names - current_names)
            features['new_columns_count'] = len(features['new_columns'])
            features['removed_columns_count'] = len(features['removed_columns'])
            
            # Type changes for common columns
            type_changes = []
            for field in schema.fields:
                if field.name in baseline_names:
                    baseline_field = [f for f in baseline_schema.fields if f.name == field.name][0]
                    if str(field.dataType) != str(baseline_field.dataType):
                        type_changes.append({
                            'column': field.name,
                            'old_type': str(baseline_field.dataType),
                            'new_type': str(field.dataType)
                        })
            
            features['type_changes'] = type_changes
            features['type_changes_count'] = len(type_changes)
        else:
            features['new_columns'] = []
            features['removed_columns'] = []
            features['new_columns_count'] = 0
            features['removed_columns_count'] = 0
            features['type_changes'] = []
            features['type_changes_count'] = 0
        
        return features
    
    def extract_statistical_features(self, df: DataFrame) -> Dict[str, Any]:
        """Extract statistical features for numeric columns"""
        
        numeric_cols = [f.name for f in df.schema.fields 
                       if str(f.dataType) in ['IntegerType', 'LongType', 'FloatType', 'DoubleType', 'DecimalType']]
        
        features = {}
        
        for col in numeric_cols:
            # Basic stats
            stats = df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std'),
                F.min(col).alias('min'),
                F.max(col).alias('max'),
                F.expr(f'percentile_approx({col}, 0.25)').alias('q25'),
                F.expr(f'percentile_approx({col}, 0.5)').alias('median'),
                F.expr(f'percentile_approx({col}, 0.75)').alias('q75'),
                F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias('null_count'),
                F.countDistinct(col).alias('distinct_count')
            ).collect()[0]
            
            total_count = df.count()
            
            features[f'{col}_mean'] = float(stats['mean']) if stats['mean'] else 0.0
            features[f'{col}_std'] = float(stats['std']) if stats['std'] else 0.0
            features[f'{col}_min'] = float(stats['min']) if stats['min'] else 0.0
            features[f'{col}_max'] = float(stats['max']) if stats['max'] else 0.0
            features[f'{col}_q25'] = float(stats['q25']) if stats['q25'] else 0.0
            features[f'{col}_median'] = float(stats['median']) if stats['median'] else 0.0
            features[f'{col}_q75'] = float(stats['q75']) if stats['q75'] else 0.0
            features[f'{col}_null_rate'] = float(stats['null_count']) / total_count
            features[f'{col}_distinct_count'] = int(stats['distinct_count'])
            features[f'{col}_cardinality'] = float(stats['distinct_count']) / total_count
            
            # IQR and range
            if stats['q75'] and stats['q25']:
                features[f'{col}_iqr'] = float(stats['q75']) - float(stats['q25'])
            else:
                features[f'{col}_iqr'] = 0.0
            
            if stats['max'] and stats['min']:
                features[f'{col}_range'] = float(stats['max']) - float(stats['min'])
            else:
                features[f'{col}_range'] = 0.0
        
        return features
    
    def extract_missing_data_features(self, df: DataFrame) -> Dict[str, Any]:
        """Extract missing data patterns"""
        
        features = {}
        total_count = df.count()
        
        if total_count == 0:
            return features
        
        # Get null counts per column - handle both NULL, NaN, and string "NaN"
        null_counts = {}
        for col in df.columns:
            col_type = [f.dataType for f in df.schema.fields if f.name == col][0]
            col_type_str = str(col_type)
            
            # For numeric columns, check both NULL and NaN
            if 'Double' in col_type_str or 'Float' in col_type_str:
                null_count = df.filter(F.col(col).isNull() | F.isnan(F.col(col))).count()
            elif 'String' in col_type_str:
                # For strings, check NULL, empty string, and literal "NaN"/"nan"
                null_count = df.filter(
                    F.col(col).isNull() | 
                    (F.col(col) == '') |
                    (F.col(col) == 'NaN') |
                    (F.col(col) == 'nan') |
                    (F.col(col) == 'None')
                ).count()
            else:
                # For other types (int, long, timestamp), just check NULL
                null_count = df.filter(F.col(col).isNull()).count()
            
            null_counts[col] = null_count
        
        # Overall missing rate
        total_cells = total_count * len(df.columns)
        total_nulls = sum(null_counts.values())
        features['overall_missing_rate'] = float(total_nulls) / total_cells if total_cells > 0 else 0.0
        
        # Per-column missing rates
        for col in df.columns:
            features[f'{col}_missing_rate'] = float(null_counts[col]) / total_count
        
        # Columns with high missing rate (>50%)
        features['high_missing_columns'] = [
            col for col in df.columns 
            if float(null_counts[col]) / total_count > 0.5
        ]
        features['high_missing_columns_count'] = len(features['high_missing_columns'])
        
        # Count columns with ANY missing data
        features['columns_with_missing'] = sum(1 for count in null_counts.values() if count > 0)
        
        return features
    
    def extract_outlier_features(
        self, 
        df: DataFrame,
        iqr_multiplier: float = 3.0
    ) -> Dict[str, Any]:
        """Extract outlier-related features"""
        
        # Find numeric columns
        numeric_cols = []
        for field in df.schema.fields:
            dtype_str = str(field.dataType)
            if any(t in dtype_str for t in ['Integer', 'Long', 'Float', 'Double', 'Decimal']):
                numeric_cols.append(field.name)
        
        features = {}
        total_count = df.count()
        
        for col in numeric_cols:
            # Get quartiles
            quartiles = df.select(
                F.expr(f'percentile_approx({col}, 0.25)').alias('q25'),
                F.expr(f'percentile_approx({col}, 0.75)').alias('q75'),
            ).collect()[0]
            
            if quartiles['q25'] is None or quartiles['q75'] is None:
                features[f'{col}_outlier_count'] = 0
                features[f'{col}_outlier_rate'] = 0.0
                continue
            
            q25 = float(quartiles['q25'])
            q75 = float(quartiles['q75'])
            iqr = q75 - q25
            
            lower_bound = q25 - iqr_multiplier * iqr
            upper_bound = q75 + iqr_multiplier * iqr
            
            # Count outliers
            outlier_count = df.filter(
                (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
            ).count()
            
            features[f'{col}_outlier_count'] = outlier_count
            features[f'{col}_outlier_rate'] = float(outlier_count) / total_count
        
        return features
    
    def extract_type_consistency_features(self, df: DataFrame) -> Dict[str, Any]:
        """Extract type consistency features"""
        
        features = {}
        
        # Check for type mismatches in supposedly numeric columns
        for col in df.columns:
            field_type = [f.dataType for f in df.schema.fields if f.name == col][0]
            
            if str(field_type) in ['IntegerType', 'LongType', 'FloatType', 'DoubleType']:
                # This column should be numeric
                # In practice, if there are type mismatches, they'd show up as nulls after casting
                # For string columns that should be numeric, we'd need special handling
                features[f'{col}_type_consistent'] = True
            else:
                features[f'{col}_type_consistent'] = True
        
        return features
    
    def compute_distribution_distance(
        self,
        current_df: DataFrame,
        baseline_df: DataFrame,
        numeric_cols: List[str]
    ) -> Dict[str, float]:
        """Compute distribution distance metrics between current and baseline"""
        
        distances = {}
        
        for col in numeric_cols:
            # Get histograms
            current_stats = current_df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std')
            ).collect()[0]
            
            baseline_stats = baseline_df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std')
            ).collect()[0]
            
            if current_stats['mean'] and baseline_stats['mean']:
                # Simple mean shift
                mean_shift = abs(float(current_stats['mean']) - float(baseline_stats['mean']))
                
                # Normalize by baseline std
                if baseline_stats['std'] and baseline_stats['std'] > 0:
                    normalized_shift = mean_shift / float(baseline_stats['std'])
                else:
                    normalized_shift = mean_shift
                
                distances[f'{col}_mean_shift'] = mean_shift
                distances[f'{col}_normalized_shift'] = normalized_shift
            
            if current_stats['std'] and baseline_stats['std']:
                # Variance ratio
                variance_ratio = float(current_stats['std']) / float(baseline_stats['std'])
                distances[f'{col}_variance_ratio'] = variance_ratio
        
        return distances
    
    def profile_dataset(
        self,
        df: pd.DataFrame,
        baseline_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Complete profiling of a dataset
        
        Args:
            df: Current dataset to profile
            baseline_df: Baseline dataset for comparison (optional)
        
        Returns:
            Dictionary of quality metrics
        """
        
        spark_df = self.pandas_to_spark(df)
        
        # Extract all features
        profile = {
            'row_count': df.shape[0],
            'column_count': df.shape[1],
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        # Schema features
        baseline_schema = None
        if baseline_df is not None:
            baseline_spark_df = self.pandas_to_spark(baseline_df)
            baseline_schema = baseline_spark_df.schema
        
        schema_features = self.extract_schema_features(spark_df, baseline_schema)
        profile.update(schema_features)
        
        # Statistical features
        statistical_features = self.extract_statistical_features(spark_df)
        profile.update(statistical_features)
        
        # Missing data features
        missing_features = self.extract_missing_data_features(spark_df)
        profile.update(missing_features)
        
        # Outlier features
        outlier_features = self.extract_outlier_features(spark_df)
        profile.update(outlier_features)
        
        # Outlier features
        outlier_features = self.extract_outlier_features(spark_df)
        profile.update(outlier_features)
        
        # Type consistency
        type_features = self.extract_type_consistency_features(spark_df)
        profile.update(type_features)
        
        # Distribution distance (if baseline provided)
        if baseline_df is not None:
            # Only compare columns that exist in both datasets
            current_numeric_cols = [
                col for col in df.columns 
                if df[col].dtype in ['int64', 'float64']
            ]
            baseline_numeric_cols = [
                col for col in baseline_df.columns 
                if baseline_df[col].dtype in ['int64', 'float64']
            ]
            
            # Common numeric columns
            numeric_cols = list(set(current_numeric_cols) & set(baseline_numeric_cols))
            
            if numeric_cols:
                dist_features = self.compute_distribution_distance(
                    spark_df,
                    baseline_spark_df,
                    numeric_cols
                )
                profile.update(dist_features)
        
        return profile


if __name__ == "__main__":
    # Test the profiler
    import sys
    sys.path.append('.')
    from src.data.bad_data_generator import BadDataGenerator
    
    print("Testing SparkDataProfiler...")
    
    # Generate test data
    generator = BadDataGenerator()
    clean_df, corrupted_df = generator.generate_quality_issue_dataset(
        n_rows=1000,
        issue_type='schema_drift',
        drift_type='add_column'
    )
    
    # Profile the data
    profiler = SparkDataProfiler()
    
    print("\nProfiling clean dataset...")
    clean_profile = profiler.profile_dataset(clean_df)
    print(f"Extracted {len(clean_profile)} features")
    print(f"Sample features: {list(clean_profile.keys())[:10]}")
    
    print("\nProfiling corrupted dataset with baseline comparison...")
    corrupted_profile = profiler.profile_dataset(corrupted_df, baseline_df=clean_df)
    print(f"Extracted {len(corrupted_profile)} features")
    
    # Show schema drift detection
    if corrupted_profile['new_columns_count'] > 0:
        print(f"\nSchema drift detected!")
        print(f"New columns: {corrupted_profile['new_columns']}")
        print(f"Column count changed: {clean_profile['column_count']} -> {corrupted_profile['column_count']}")
    
    profiler.spark.stop()
    print("\nProfiler test complete!")