"""
Generate synthetic datasets with intentional data quality issues
for training the quality detectors
"""

import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta


class BadDataGenerator:
    """Generate datasets with various quality issues"""
    
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_clean_dataset(
        self, 
        n_rows: int = 10000,
        n_features: int = 10
    ) -> pd.DataFrame:
        """Generate a clean baseline dataset"""
        
        data = {
            'id': range(n_rows),
            'timestamp': [
                datetime.now() - timedelta(days=random.randint(0, 365))
                for _ in range(n_rows)
            ],
            'user_id': [self.fake.uuid4() for _ in range(n_rows)],
            'name': [self.fake.name() for _ in range(n_rows)],
            'email': [self.fake.email() for _ in range(n_rows)],
            'age': np.random.randint(18, 80, n_rows),
            'salary': np.random.normal(60000, 20000, n_rows),
            'department': np.random.choice(
                ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
                n_rows
            ),
            'experience_years': np.random.randint(0, 30, n_rows),
            'performance_score': np.random.uniform(1.0, 5.0, n_rows),
        }
        
        # Add correlation: salary increases with experience
        data['salary'] = data['salary'] + data['experience_years'] * 1500
        
        return pd.DataFrame(data)
    
    def inject_schema_drift(
        self,
        df: pd.DataFrame,
        drift_type: str = 'add_column',
        subtlety: float = 0.5
    ) -> pd.DataFrame:
        """
        Inject schema drift issues with varying subtlety
        
        Args:
            subtlety: 0.0 (very obvious) to 1.0 (very subtle)
        """
        
        df = df.copy()
        
        if drift_type == 'add_column':
            # Add 1-3 columns based on subtlety
            num_cols = 1 if subtlety > 0.7 else (2 if subtlety > 0.3 else 3)
            for i in range(num_cols):
                col_name = f'new_field_{i}' if num_cols > 1 else 'new_field'
                df[col_name] = np.random.randint(0, 100, len(df))
            
        elif drift_type == 'remove_column':
            # Remove 1-2 columns
            cols_to_remove = ['performance_score']
            if subtlety < 0.5 and 'experience_years' in df.columns:
                cols_to_remove.append('experience_years')
            
            for col in cols_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                
        elif drift_type == 'type_change':
            # Column type changes - make it subtle
            if subtlety > 0.5:
                # Subtle: only affect some rows
                mask = np.random.random(len(df)) < 0.3
                df.loc[mask, 'age'] = df.loc[mask, 'age'].astype(str)
            else:
                # Obvious: change entire column
                df['age'] = df['age'].astype(str)
            
        elif drift_type == 'rename_column':
            # Column gets renamed
            if subtlety > 0.5:
                df = df.rename(columns={'salary': 'annual_salary'})
            else:
                df = df.rename(columns={
                    'salary': 'annual_salary',
                    'experience_years': 'years_experience'
                })
            
        return df
    
    def inject_distribution_shift(
        self,
        df: pd.DataFrame,
        column: str = 'salary',
        shift_factor: float = 2.0
    ) -> pd.DataFrame:
        """Inject distribution shift"""
        
        df = df.copy()
        
        if column in df.columns:
            # Shift the distribution
            df[column] = df[column] * shift_factor + np.random.normal(0, 5000, len(df))
        
        return df
    
    def inject_missing_data(
        self,
        df: pd.DataFrame,
        missing_rate: float = 0.3,
        pattern: str = 'random'
    ) -> pd.DataFrame:
        """Inject missing data patterns"""
        
        df = df.copy()
        
        if pattern == 'random':
            # Random missingness
            for col in ['email', 'salary', 'performance_score']:
                if col in df.columns:
                    mask = np.random.random(len(df)) < missing_rate
                    df.loc[mask, col] = np.nan
                    
        elif pattern == 'systematic':
            # Systematic missingness (e.g., all salaries missing for HR)
            if 'department' in df.columns and 'salary' in df.columns:
                mask = df['department'] == 'HR'
                df.loc[mask, 'salary'] = np.nan
                
        elif pattern == 'streak':
            # Consecutive missing values (data pipeline failure)
            start_idx = len(df) // 2
            end_idx = start_idx + int(len(df) * missing_rate)
            df.loc[start_idx:end_idx, 'email'] = np.nan
            
        return df
    
    def inject_outliers(
        self,
        df: pd.DataFrame,
        column: str = 'salary',
        outlier_rate: float = 0.05
    ) -> pd.DataFrame:
        """Inject outlier values"""
        
        df = df.copy()
        
        if column not in df.columns:
            return df
        
        n_outliers = int(len(df) * outlier_rate)
        
        if n_outliers == 0:
            return df
        
        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
        
        original_max = df[column].max()
        
        # Generate GUARANTEED extreme outliers
        outlier_values = []
        for i in range(n_outliers):
            if np.random.random() < 0.5:
                val = original_max * 100
            else:
                val = -abs(original_max) * 50
            
            outlier_values.append(val)
        
        df.loc[outlier_indices, column] = outlier_values
        
        return df
    
    def inject_type_mismatches(
        self,
        df: pd.DataFrame,
        error_rate: float = 0.1
    ) -> pd.DataFrame:
        """Inject data type mismatches"""
        
        df = df.copy()
        
        # Inject string values in numeric columns
        if 'age' in df.columns:
            n_errors = int(len(df) * error_rate)
            error_indices = np.random.choice(len(df), n_errors, replace=False)
            df.loc[error_indices, 'age'] = 'unknown'
            
        # Inject invalid email formats
        if 'email' in df.columns:
            n_errors = int(len(df) * error_rate)
            error_indices = np.random.choice(len(df), n_errors, replace=False)
            df.loc[error_indices, 'email'] = 'not_an_email'
            
        return df
    
    def inject_correlation_break(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Break correlations between features"""
        
        df = df.copy()
        
        # Break salary-experience correlation
        if 'salary' in df.columns and 'experience_years' in df.columns:
            # Randomize salary (no longer correlated with experience)
            df['salary'] = np.random.normal(60000, 20000, len(df))
            
        return df
    
    def generate_quality_issue_dataset(
        self,
        n_rows: int = 10000,
        issue_type: str = 'schema_drift',
        subtlety: float = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a pair of datasets: clean baseline and one with quality issues
        
        Args:
            subtlety: 0.0-1.0, how subtle the issue should be (random if None)
        
        Returns:
            Tuple of (clean_df, corrupted_df)
        """
        
        # Random subtlety if not specified
        if subtlety is None:
            subtlety = np.random.uniform(0.2, 0.9)
        
        clean_df = self.generate_clean_dataset(n_rows)
        
        if issue_type == 'schema_drift':
            corrupted_df = self.inject_schema_drift(
                clean_df,
                kwargs.get('drift_type', np.random.choice([
                    'add_column', 'remove_column', 'type_change', 'rename_column'
                ])),
                subtlety=subtlety
            )
        elif issue_type == 'distribution_shift':
            corrupted_df = self.inject_distribution_shift(
                clean_df,
                kwargs.get('column', 'salary'),
                kwargs.get('shift_factor', 2.0)
            )
        elif issue_type == 'missing_data':
            corrupted_df = self.inject_missing_data(
                clean_df,
                kwargs.get('missing_rate', 0.3),
                kwargs.get('pattern', 'random')
            )
        elif issue_type == 'outliers' or issue_type == 'outlier':
            corrupted_df = self.inject_outliers(
                clean_df,
                kwargs.get('column', 'salary'),
                kwargs.get('outlier_rate', 0.05)
            )
        elif issue_type == 'type_mismatch':
            corrupted_df = self.inject_type_mismatches(
                clean_df,
                kwargs.get('error_rate', 0.1)
            )
        elif issue_type == 'correlation_break':
            corrupted_df = self.inject_correlation_break(clean_df)
        else:
            corrupted_df = clean_df.copy()
            
        return clean_df, corrupted_df
    
    def generate_training_dataset(
        self,
        n_samples_per_issue: int = 1000,
        save_path: str = None
    ) -> Dict[str, List[Tuple[pd.DataFrame, pd.DataFrame, str]]]:
        """
        Generate comprehensive training dataset with all issue types
        
        Returns:
            Dictionary mapping issue types to list of (clean, corrupted, label) tuples
        """
        
        issue_types = [
            'schema_drift',
            'distribution_shift',
            'missing_data',
            'outliers',
            'type_mismatch',
            'correlation_break'
        ]
        
        training_data = {issue: [] for issue in issue_types}
        training_data['clean'] = []
        
        print(f"Generating training data...")
        
        # Generate clean samples
        for i in range(n_samples_per_issue):
            clean_df = self.generate_clean_dataset(n_rows=1000)
            training_data['clean'].append((clean_df, clean_df, 'clean'))
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples_per_issue} clean samples")
        
        # Generate samples with issues
        for issue_type in issue_types:
            print(f"Generating {issue_type} samples...")
            
            for i in range(n_samples_per_issue):
                clean_df, corrupted_df = self.generate_quality_issue_dataset(
                    n_rows=1000,
                    issue_type=issue_type
                )
                training_data[issue_type].append((clean_df, corrupted_df, issue_type))
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{n_samples_per_issue} {issue_type} samples")
        
        # Save if path provided
        if save_path:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(training_data, f)
            print(f"\nTraining data saved to {save_path}")
        
        return training_data


if __name__ == "__main__":
    # Example usage
    generator = BadDataGenerator()
    
    # Generate a single example
    print("Generating example datasets...")
    clean_df, corrupted_df = generator.generate_quality_issue_dataset(
        n_rows=100,
        issue_type='schema_drift',
        drift_type='add_column'
    )
    
    print("\nClean dataset:")
    print(clean_df.head())
    print(f"Shape: {clean_df.shape}")
    print(f"Columns: {clean_df.columns.tolist()}")
    
    print("\nCorrupted dataset (schema drift):")
    print(corrupted_df.head())
    print(f"Shape: {corrupted_df.shape}")
    print(f"Columns: {corrupted_df.columns.tolist()}")
    
    print("\nTest complete!")