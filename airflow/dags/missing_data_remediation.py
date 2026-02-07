"""
Airflow DAG for Missing Data Remediation
Automatically imputes missing values
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

default_args = {
    'owner': 'data-quality',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

DATA_DIR = '/opt/airflow/data/pipelines'


def analyze_missing_patterns(**context):
    """Analyze missing data patterns"""
    
    task_instance = context['task_instance']
    dag_run = context['dag_run']
    
    pipeline_id = dag_run.conf.get('pipeline_id', 'unknown')
    data_file = dag_run.conf.get('data_file')
    
    print(f"Analyzing missing data for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    # Analyze missing patterns
    missing_info = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing_info[col] = {
                'count': int(null_count),
                'rate': float(null_count / len(df))
            }
    
    print(f"  Columns with missing data: {len(missing_info)}")
    for col, info in missing_info.items():
        print(f"    {col}: {info['count']} ({info['rate']:.2%})")
    
    task_instance.xcom_push(key='missing_info', value={
        'pipeline_id': pipeline_id,
        'data_file': data_file,
        'missing_columns': missing_info,
        'total_missing_rate': float(df.isnull().sum().sum() / (len(df) * len(df.columns)))
    })
    
    return missing_info


def impute_missing_values(**context):
    """Impute missing values using appropriate strategy"""
    
    task_instance = context['task_instance']
    missing_info = task_instance.xcom_pull(key='missing_info')
    
    pipeline_id = missing_info['pipeline_id']
    data_file = missing_info['data_file']
    
    print(f"Imputing missing values for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Impute numeric columns with KNN
    if numeric_cols:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"  Imputed {len(numeric_cols)} numeric columns with KNN")
    
    # Impute categorical with mode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'UNKNOWN'
            df[col].fillna(mode_value, inplace=True)
    
    if categorical_cols:
        print(f"  Imputed {len(categorical_cols)} categorical columns with mode")
    
    # Save imputed data
    output_file = f"{DATA_DIR}/{pipeline_id}_imputed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"  Saved imputed data: {output_file}")
    
    task_instance.xcom_push(key='imputed_file', value=output_file)
    
    return output_file


def validate_imputation(**context):
    """Validate that imputation was successful"""
    
    task_instance = context['task_instance']
    imputed_file = task_instance.xcom_pull(key='imputed_file')
    missing_info = task_instance.xcom_pull(key='missing_info')
    
    print("Validating imputation...")
    
    df = pd.read_csv(imputed_file)
    
    # Check if missing values are gone
    remaining_nulls = df.isnull().sum().sum()
    original_missing_rate = missing_info['total_missing_rate']
    
    validation_result = {
        'success': remaining_nulls == 0,
        'remaining_nulls': int(remaining_nulls),
        'original_missing_rate': original_missing_rate,
        'new_missing_rate': float(remaining_nulls / (len(df) * len(df.columns))),
        'issues_resolved': ['missing_data'] if remaining_nulls == 0 else []
    }
    
    if remaining_nulls == 0:
        print(f"  ✓ All missing values imputed")
    else:
        print(f"  ✗ {remaining_nulls} missing values remain")
    
    task_instance.xcom_push(key='validation', value=validation_result)
    
    return validation_result


def send_notification(**context):
    """Send completion notification"""
    
    task_instance = context['task_instance']
    validation = task_instance.xcom_pull(key='validation')
    missing_info = task_instance.xcom_pull(key='missing_info')
    
    pipeline_id = missing_info['pipeline_id']
    
    print(f"Missing data remediation complete for: {pipeline_id}")
    
    if validation['success']:
        print("STATUS: SUCCESS")
        print(f"  Missing rate: {validation['original_missing_rate']:.2%} → {validation['new_missing_rate']:.2%}")
    else:
        print("STATUS: PARTIAL SUCCESS")
        print(f"  {validation['remaining_nulls']} missing values remain")
    
    return validation


dag = DAG(
    'missing_data_remediation',
    default_args=default_args,
    description='Automatically impute missing data values',
    schedule_interval=None,
    catchup=False,
    tags=['data-quality', 'auto-remediation', 'missing-data']
)


analyze = PythonOperator(
    task_id='analyze_missing_patterns',
    python_callable=analyze_missing_patterns,
    dag=dag
)

impute = PythonOperator(
    task_id='impute_missing_values',
    python_callable=impute_missing_values,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_imputation',
    python_callable=validate_imputation,
    dag=dag
)

notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

analyze >> impute >> validate >> notify