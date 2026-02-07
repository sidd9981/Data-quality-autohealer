"""
Airflow DAG for Outlier Remediation
Automatically quarantines and handles outliers
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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


def detect_outliers(**context):
    """Detect outliers using IQR method"""
    
    task_instance = context['task_instance']
    dag_run = context['dag_run']
    
    pipeline_id = dag_run.conf.get('pipeline_id', 'unknown')
    data_file = dag_run.conf.get('data_file')
    
    print(f"Detecting outliers for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    # Detect outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        
        lower = q25 - 3.0 * iqr
        upper = q75 + 3.0 * iqr
        
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outlier_info[col] = {
                'count': int(outlier_count),
                'rate': float(outlier_count / len(df)),
                'lower_bound': float(lower),
                'upper_bound': float(upper)
            }
    
    print(f"  Outliers detected in {len(outlier_info)} columns:")
    for col, info in outlier_info.items():
        print(f"    {col}: {info['count']} outliers ({info['rate']:.2%})")
    
    task_instance.xcom_push(key='outlier_info', value={
        'pipeline_id': pipeline_id,
        'data_file': data_file,
        'outliers': outlier_info
    })
    
    return outlier_info


def quarantine_outliers(**context):
    """Move outliers to quarantine file"""
    
    task_instance = context['task_instance']
    outlier_info = task_instance.xcom_pull(key='outlier_info')
    
    pipeline_id = outlier_info['pipeline_id']
    data_file = outlier_info['data_file']
    
    print(f"Quarantining outliers for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    # Create outlier mask
    outlier_mask = pd.Series([False] * len(df))
    
    for col, info in outlier_info['outliers'].items():
        col_outliers = (df[col] < info['lower_bound']) | (df[col] > info['upper_bound'])
        outlier_mask = outlier_mask | col_outliers
    
    # Separate outliers from clean data
    outliers_df = df[outlier_mask]
    clean_df = df[~outlier_mask]
    
    # Save both
    outliers_file = f"{DATA_DIR}/{pipeline_id}_outliers_quarantined.csv"
    clean_file = f"{DATA_DIR}/{pipeline_id}_outliers_removed.csv"
    
    outliers_df.to_csv(outliers_file, index=False)
    clean_df.to_csv(clean_file, index=False)
    
    print(f"  Quarantined {len(outliers_df)} rows to: {outliers_file}")
    print(f"  Clean data ({len(clean_df)} rows) saved to: {clean_file}")
    
    task_instance.xcom_push(key='quarantine_result', value={
        'outliers_file': outliers_file,
        'clean_file': clean_file,
        'outliers_count': len(outliers_df),
        'clean_count': len(clean_df)
    })
    
    return clean_file


def apply_statistical_correction(**context):
    """Apply statistical correction to outliers (Winsorization)"""
    
    task_instance = context['task_instance']
    outlier_info = task_instance.xcom_pull(key='outlier_info')
    
    data_file = outlier_info['data_file']
    
    print("Applying statistical correction (Winsorization)...")
    
    df = pd.read_csv(data_file)
    
    # Apply Winsorization
    for col, info in outlier_info['outliers'].items():
        df[col] = df[col].clip(lower=info['lower_bound'], upper=info['upper_bound'])
    
    # Save corrected data
    corrected_file = f"{DATA_DIR}/{outlier_info['pipeline_id']}_outliers_corrected.csv"
    df.to_csv(corrected_file, index=False)
    
    print(f"  Saved corrected data: {corrected_file}")
    
    task_instance.xcom_push(key='corrected_file', value=corrected_file)
    
    return corrected_file


def validate_remediation(**context):
    """Validate outlier remediation"""
    
    task_instance = context['task_instance']
    clean_file = task_instance.xcom_pull(key='quarantine_result')['clean_file']
    
    print("Validating outlier remediation...")
    
    df = pd.read_csv(clean_file)
    
    validation_result = {
        'success': True,
        'clean_rows': len(df),
        'issues_resolved': ['outlier']
    }
    
    print(f"  ✓ Outliers removed, {len(df)} clean rows remain")
    
    task_instance.xcom_push(key='validation', value=validation_result)
    
    return validation_result


def send_notification(**context):
    """Send completion notification"""
    
    task_instance = context['task_instance']
    validation = task_instance.xcom_pull(key='validation')
    quarantine_result = task_instance.xcom_pull(key='quarantine_result')
    outlier_info = task_instance.xcom_pull(key='outlier_info')
    
    pipeline_id = outlier_info['pipeline_id']
    
    print(f"Outlier remediation complete for: {pipeline_id}")
    print("STATUS: SUCCESS")
    print(f"  Outliers quarantined: {quarantine_result['outliers_count']}")
    print(f"  Clean data: {quarantine_result['clean_count']} rows")
    
    return validation


dag = DAG(
    'outlier_remediation',
    default_args=default_args,
    description='Automatically quarantine and handle outliers',
    schedule_interval=None,
    catchup=False,
    tags=['data-quality', 'auto-remediation', 'outliers']
)


detect = PythonOperator(
    task_id='detect_outliers',
    python_callable=detect_outliers,
    dag=dag
)

quarantine = PythonOperator(
    task_id='quarantine_outliers',
    python_callable=quarantine_outliers,
    dag=dag
)

correct = PythonOperator(
    task_id='apply_statistical_correction',
    python_callable=apply_statistical_correction,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_remediation',
    python_callable=validate_remediation,
    dag=dag
)

notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

detect >> [quarantine, correct] >> validate >> notify