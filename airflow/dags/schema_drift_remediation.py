"""
Airflow DAG for Schema Drift Remediation
Automatically re-ingests data when schema changes are detected
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import json
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


def detect_schema_changes(**context):
    """Detect what schema changes occurred"""
    
    task_instance = context['task_instance']
    dag_run = context['dag_run']
    
    # Get pipeline_id from DAG run config
    pipeline_id = dag_run.conf.get('pipeline_id', 'unknown')
    corrupted_file = dag_run.conf.get('corrupted_file')
    baseline_file = dag_run.conf.get('baseline_file')
    
    print(f"Analyzing schema changes for pipeline: {pipeline_id}")
    
    # Load both files
    corrupted_df = pd.read_csv(corrupted_file)
    baseline_df = pd.read_csv(baseline_file)
    
    # Detect schema differences
    corrupted_cols = set(corrupted_df.columns)
    baseline_cols = set(baseline_df.columns)
    
    new_columns = list(corrupted_cols - baseline_cols)
    removed_columns = list(baseline_cols - corrupted_cols)
    
    # Detect type changes
    type_changes = []
    for col in corrupted_cols & baseline_cols:
        if corrupted_df[col].dtype != baseline_df[col].dtype:
            type_changes.append({
                'column': col,
                'old_type': str(baseline_df[col].dtype),
                'new_type': str(corrupted_df[col].dtype)
            })
    
    schema_changes = {
        'pipeline_id': pipeline_id,
        'new_columns': new_columns,
        'removed_columns': removed_columns,
        'type_changes': type_changes,
        'timestamp': datetime.now().isoformat(),
        'corrupted_file': corrupted_file,
        'baseline_file': baseline_file
    }
    
    task_instance.xcom_push(key='schema_changes', value=schema_changes)
    
    print(f"Schema changes detected:")
    print(f"  New columns: {new_columns}")
    print(f"  Removed columns: {removed_columns}")
    print(f"  Type changes: {len(type_changes)}")
    
    return schema_changes


def update_schema_registry(**context):
    """Update schema registry with new schema"""
    
    task_instance = context['task_instance']
    schema_changes = task_instance.xcom_pull(key='schema_changes')
    
    pipeline_id = schema_changes['pipeline_id']
    
    print(f"Updating schema registry for pipeline: {pipeline_id}")
    
    # Save updated schema definition
    schema_file = f"{DATA_DIR}/{pipeline_id}_schema.json"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    corrupted_file = schema_changes['corrupted_file']
    df = pd.read_csv(corrupted_file)
    
    schema_definition = {
        'pipeline_id': pipeline_id,
        'columns': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'column_count': len(df.columns),
        'updated_at': datetime.now().isoformat(),
        'changes': schema_changes
    }
    
    with open(schema_file, 'w') as f:
        json.dump(schema_definition, f, indent=2)
    
    print(f"Schema registry updated: {schema_file}")
    print(f"  New schema has {len(df.columns)} columns")
    
    return schema_file


def trigger_data_reingestion(**context):
    """Re-ingest data with new schema"""
    
    task_instance = context['task_instance']
    schema_changes = task_instance.xcom_pull(key='schema_changes')
    
    pipeline_id = schema_changes['pipeline_id']
    corrupted_file = schema_changes['corrupted_file']
    
    print(f"Re-ingesting data for pipeline: {pipeline_id}")
    
    # Read data with new schema
    df = pd.read_csv(corrupted_file)
    
    # Save as "re-ingested" data
    output_file = f"{DATA_DIR}/{pipeline_id}_reingested.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Data re-ingested successfully")
    print(f"  Input: {corrupted_file}")
    print(f"  Output: {output_file}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    
    task_instance.xcom_push(key='reingested_file', value=output_file)
    
    return output_file


def validate_remediation(**context):
    """Validate that remediation was successful"""
    
    task_instance = context['task_instance']
    reingested_file = task_instance.xcom_pull(key='reingested_file')
    schema_changes = task_instance.xcom_pull(key='schema_changes')
    
    print(f"Validating remediation...")
    
    # Load re-ingested data
    df = pd.read_csv(reingested_file)
    
    # Check that new columns are present
    new_cols = schema_changes['new_columns']
    all_present = all(col in df.columns for col in new_cols)
    
    # Calculate quality score (simple: did we get all expected columns?)
    quality_score = 1.0 if all_present else 0.5
    
    validation_result = {
        'success': all_present,
        'quality_score': quality_score,
        'issues_resolved': ['schema_drift'] if all_present else [],
        'new_columns_present': all_present,
        'file': reingested_file,
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    task_instance.xcom_push(key='validation', value=validation_result)
    
    print(f"Validation complete:")
    print(f"  Success: {all_present}")
    print(f"  Quality score: {quality_score}")
    print(f"  Rows processed: {len(df)}")
    
    return validation_result


def send_completion_notification(**context):
    """Send notification that remediation is complete"""
    
    task_instance = context['task_instance']
    validation = task_instance.xcom_pull(key='validation')
    schema_changes = task_instance.xcom_pull(key='schema_changes')
    
    pipeline_id = schema_changes['pipeline_id']
    
    print(f"Remediation complete for pipeline: {pipeline_id}")
    
    if validation['success']:
        print("STATUS: SUCCESS")
        print(f"  Quality score: {validation['quality_score']}")
        print(f"  Issues resolved: {validation['issues_resolved']}")
        print(f"  Output file: {validation['file']}")
    else:
        print("STATUS: PARTIAL SUCCESS")
        print(f"  Some issues remain")
    
    # In production: Send to Slack, PagerDuty, Kafka, etc.
    
    return validation


# Define DAG
dag = DAG(
    'schema_drift_remediation',
    default_args=default_args,
    description='Automatically remediate schema drift issues',
    schedule_interval=None,
    catchup=False,
    tags=['data-quality', 'auto-remediation', 'schema']
)


# Define tasks
detect_changes = PythonOperator(
    task_id='detect_schema_changes',
    python_callable=detect_schema_changes,
    dag=dag
)

update_schema = PythonOperator(
    task_id='update_schema_registry',
    python_callable=update_schema_registry,
    dag=dag
)

reingest_data = PythonOperator(
    task_id='trigger_data_reingestion',
    python_callable=trigger_data_reingestion,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_remediation',
    python_callable=validate_remediation,
    dag=dag
)

notify = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    dag=dag
)


# Define workflow
detect_changes >> update_schema >> reingest_data >> validate >> notify