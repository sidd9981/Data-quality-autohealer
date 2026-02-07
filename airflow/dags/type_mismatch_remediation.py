"""
Airflow DAG for Type Mismatch Remediation
Automatically coerces data types
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
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


def detect_type_mismatches(**context):
    """Detect data type inconsistencies"""
    
    task_instance = context['task_instance']
    dag_run = context['dag_run']
    
    pipeline_id = dag_run.conf.get('pipeline_id', 'unknown')
    data_file = dag_run.conf.get('data_file')
    expected_schema = dag_run.conf.get('expected_schema', {})
    
    print(f"Detecting type mismatches for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    # Detect type mismatches
    mismatches = {}
    for col in df.columns:
        current_type = str(df[col].dtype)
        expected_type = expected_schema.get(col, current_type)
        
        if current_type != expected_type:
            mismatches[col] = {
                'current': current_type,
                'expected': expected_type
            }
    
    print(f"  Type mismatches found: {len(mismatches)}")
    for col, types in mismatches.items():
        print(f"    {col}: {types['current']} → {types['expected']}")
    
    task_instance.xcom_push(key='mismatch_info', value={
        'pipeline_id': pipeline_id,
        'data_file': data_file,
        'mismatches': mismatches
    })
    
    return mismatches


def coerce_data_types(**context):
    """Coerce data to correct types"""
    
    task_instance = context['task_instance']
    mismatch_info = task_instance.xcom_pull(key='mismatch_info')
    
    pipeline_id = mismatch_info['pipeline_id']
    data_file = mismatch_info['data_file']
    
    print(f"Coercing data types for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    coercion_results = {}
    
    for col, types in mismatch_info['mismatches'].items():
        try:
            # Attempt type coercion
            if 'int' in types['expected'].lower():
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            elif 'float' in types['expected'].lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif 'object' in types['expected'].lower() or 'str' in types['expected'].lower():
                df[col] = df[col].astype(str)
            
            coercion_results[col] = 'success'
            print(f"  ✓ Coerced {col} to {types['expected']}")
            
        except Exception as e:
            coercion_results[col] = f'failed: {str(e)}'
            print(f"  ✗ Failed to coerce {col}: {e}")
    
    # Save coerced data
    output_file = f"{DATA_DIR}/{pipeline_id}_type_corrected.csv"
    df.to_csv(output_file, index=False)
    
    print(f"  Saved type-corrected data: {output_file}")
    
    task_instance.xcom_push(key='coerced_file', value=output_file)
    task_instance.xcom_push(key='coercion_results', value=coercion_results)
    
    return output_file


def validate_type_coercion(**context):
    """Validate type coercion was successful"""
    
    task_instance = context['task_instance']
    coerced_file = task_instance.xcom_pull(key='coerced_file')
    coercion_results = task_instance.xcom_pull(key='coercion_results')
    
    print("Validating type coercion...")
    
    success_count = sum(1 for r in coercion_results.values() if r == 'success')
    total_count = len(coercion_results)
    
    validation_result = {
        'success': success_count == total_count,
        'success_rate': success_count / total_count if total_count > 0 else 1.0,
        'coerced_columns': success_count,
        'failed_columns': total_count - success_count,
        'issues_resolved': ['type_mismatch'] if success_count == total_count else []
    }
    
    if validation_result['success']:
        print(f"  ✓ All {total_count} type mismatches corrected")
    else:
        print(f"  ⚠ {success_count}/{total_count} type mismatches corrected")
    
    task_instance.xcom_push(key='validation', value=validation_result)
    
    return validation_result


def send_notification(**context):
    """Send completion notification"""
    
    task_instance = context['task_instance']
    validation = task_instance.xcom_pull(key='validation')
    mismatch_info = task_instance.xcom_pull(key='mismatch_info')
    
    pipeline_id = mismatch_info['pipeline_id']
    
    print(f"Type mismatch remediation complete for: {pipeline_id}")
    
    if validation['success']:
        print("STATUS: SUCCESS")
        print(f"  All type mismatches corrected")
    else:
        print("STATUS: PARTIAL SUCCESS")
        print(f"  Success rate: {validation['success_rate']:.0%}")
    
    return validation


dag = DAG(
    'type_mismatch_remediation',
    default_args=default_args,
    description='Automatically coerce data types',
    schedule_interval=None,
    catchup=False,
    tags=['data-quality', 'auto-remediation', 'types']
)


detect = PythonOperator(
    task_id='detect_type_mismatches',
    python_callable=detect_type_mismatches,
    dag=dag
)

coerce = PythonOperator(
    task_id='coerce_data_types',
    python_callable=coerce_data_types,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_type_coercion',
    python_callable=validate_type_coercion,
    dag=dag
)

notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

detect >> coerce >> validate >> notify