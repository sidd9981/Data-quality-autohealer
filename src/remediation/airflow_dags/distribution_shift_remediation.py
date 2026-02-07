"""
Airflow DAG for Distribution Shift Remediation
Automatically retrains ML models when data distribution changes
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
MODEL_DIR = '/opt/airflow/models/production'


def collect_new_data(**context):
    """Collect data with new distribution"""
    
    task_instance = context['task_instance']
    dag_run = context['dag_run']
    
    pipeline_id = dag_run.conf.get('pipeline_id', 'unknown')
    data_file = dag_run.conf.get('data_file')
    
    print(f"Collecting new data for pipeline: {pipeline_id}")
    
    df = pd.read_csv(data_file)
    
    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    
    task_instance.xcom_push(key='data_info', value={
        'pipeline_id': pipeline_id,
        'data_file': data_file,
        'row_count': len(df),
        'column_count': len(df.columns)
    })
    
    return data_file


def retrain_model(**context):
    """Retrain ML model on new data distribution"""
    
    task_instance = context['task_instance']
    data_info = task_instance.xcom_pull(key='data_info')
    
    pipeline_id = data_info['pipeline_id']
    data_file = data_info['data_file']
    
    print(f"Retraining model for pipeline: {pipeline_id}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Simple example: train classifier on salary prediction
    if 'salary' in df.columns and 'experience_years' in df.columns:
        X = df[['experience_years', 'age']].fillna(0)
        y = (df['salary'] > df['salary'].median()).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file = f"{MODEL_DIR}/{pipeline_id}_model.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  Model trained and saved: {model_file}")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        
        task_instance.xcom_push(key='model_info', value={
            'model_file': model_file,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'samples_trained': len(X_train)
        })
        
        return model_file
    else:
        print("  Insufficient columns for training, skipping")
        return None


def validate_model(**context):
    """Validate retrained model performance"""
    
    task_instance = context['task_instance']
    model_info = task_instance.xcom_pull(key='model_info')
    
    if not model_info:
        print("No model to validate")
        return False
    
    print("Validating retrained model...")
    
    # Check if accuracy is acceptable
    test_acc = model_info['test_accuracy']
    
    if test_acc > 0.7:
        print(f"  ✓ Model validation passed (accuracy: {test_acc:.4f})")
        validation_result = {
            'success': True,
            'quality_score': test_acc,
            'issues_resolved': ['distribution_shift']
        }
    else:
        print(f"  ✗ Model validation failed (accuracy: {test_acc:.4f})")
        validation_result = {
            'success': False,
            'quality_score': test_acc,
            'issues_resolved': []
        }
    
    task_instance.xcom_push(key='validation', value=validation_result)
    return validation_result


def deploy_model(**context):
    """Deploy retrained model to production"""
    
    task_instance = context['task_instance']
    model_info = task_instance.xcom_pull(key='model_info')
    validation = task_instance.xcom_pull(key='validation')
    
    if not validation['success']:
        print("Validation failed, not deploying model")
        return False
    
    model_file = model_info['model_file']
    
    print(f"Deploying model to production...")
    print(f"  Model: {model_file}")
    print(f"  Accuracy: {model_info['test_accuracy']:.4f}")
    
    # In production: Update serving endpoint, A/B test, gradual rollout
    # For demo: Just mark as deployed
    
    print("  ✓ Model deployed successfully")
    
    return True


def send_notification(**context):
    """Send completion notification"""
    
    task_instance = context['task_instance']
    validation = task_instance.xcom_pull(key='validation')
    data_info = task_instance.xcom_pull(key='data_info')
    
    pipeline_id = data_info['pipeline_id']
    
    print(f"Distribution shift remediation complete for: {pipeline_id}")
    
    if validation['success']:
        print("STATUS: SUCCESS")
        print(f"  Model retrained and deployed")
        print(f"  New accuracy: {validation['quality_score']:.4f}")
    else:
        print("STATUS: FAILED")
        print("  Model retraining did not meet quality threshold")
    
    return validation


dag = DAG(
    'distribution_shift_remediation',
    default_args=default_args,
    description='Automatically retrain models when distribution shifts',
    schedule_interval=None,
    catchup=False,
    tags=['data-quality', 'auto-remediation', 'distribution']
)


collect_data = PythonOperator(
    task_id='collect_new_data',
    python_callable=collect_new_data,
    dag=dag
)

retrain = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

collect_data >> retrain >> validate >> deploy >> notify