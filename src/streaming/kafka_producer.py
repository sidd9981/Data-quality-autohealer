"""
Kafka Producer for streaming quality metrics
"""

from confluent_kafka import Producer
import json
from typing import Dict
import pandas as pd
from datetime import datetime
import sys
sys.path.append('.')


class QualityMetricsProducer:
    """Stream quality metrics to Kafka"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9093'):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'quality-metrics-producer'
        })
        
        self.topics = {
            'metrics': 'data-quality-metrics',
            'alerts': 'quality-alerts',
            'actions': 'remediation-actions'
        }
    
    def delivery_callback(self, err, msg):
        """Callback for message delivery"""
        if err:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def send_quality_metrics(
        self,
        pipeline_id: str,
        profile: Dict,
        partition_key: str = None
    ):
        """Send quality profile to Kafka"""
        
        message = {
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': profile
        }
        
        self.producer.produce(
            self.topics['metrics'],
            key=(partition_key or pipeline_id).encode('utf-8'),
            value=json.dumps(message).encode('utf-8'),
            callback=self.delivery_callback
        )
        
        self.producer.flush()
    
    def send_quality_alert(
        self,
        pipeline_id: str,
        issue_types: list,
        severity: str = 'medium'
    ):
        """Send quality alert to Kafka"""
        
        message = {
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'issue_types': issue_types,
            'severity': severity
        }
        
        self.producer.produce(
            self.topics['alerts'],
            key=pipeline_id.encode('utf-8'),
            value=json.dumps(message).encode('utf-8'),
            callback=self.delivery_callback
        )
        
        self.producer.flush()
    
    def send_remediation_action(
        self,
        pipeline_id: str,
        issue_type: str,
        action_type: str,
        status: str,
        details: Dict = None
    ):
        """Send remediation action event to Kafka"""
        
        message = {
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'issue_type': issue_type,
            'action_type': action_type,
            'status': status,
            'details': details or {}
        }
        
        self.producer.produce(
            self.topics['actions'],
            key=pipeline_id.encode('utf-8'),
            value=json.dumps(message).encode('utf-8'),
            callback=self.delivery_callback
        )
        
        self.producer.flush()
    
    def close(self):
        """Close producer"""
        self.producer.flush()


if __name__ == "__main__":
    from src.data.bad_data_generator import BadDataGenerator
    from src.profilers.spark_profiler import SparkDataProfiler
    
    print("Testing Kafka Producer...")
    
    # Create producer
    producer = QualityMetricsProducer()
    
    # Generate test data
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    
    # Test 1: Send quality metrics
    print("\n1. Sending quality metrics...")
    clean_df, drift_df = generator.generate_quality_issue_dataset(
        n_rows=200,
        issue_type='schema_drift'
    )
    
    profile = profiler.profile_dataset(drift_df, baseline_df=clean_df)
    
    producer.send_quality_metrics(
        pipeline_id='test_pipeline_1',
        profile=profile
    )
    print("  Sent quality metrics to Kafka")
    
    # Test 2: Send alert
    print("\n2. Sending quality alert...")
    producer.send_quality_alert(
        pipeline_id='test_pipeline_1',
        issue_types=['schema_drift'],
        severity='high'
    )
    print("  Sent alert to Kafka")
    
    # Test 3: Send remediation action
    print("\n3. Sending remediation action...")
    producer.send_remediation_action(
        pipeline_id='test_pipeline_1',
        issue_type='schema_drift',
        action_type='re_ingest',
        status='started',
        details={'new_columns': ['field_1', 'field_2']}
    )
    print("  Sent remediation action to Kafka")
    
    producer.close()
    profiler.spark.stop()
    
    print("\nKafka Producer test complete!")
    print("\nTo verify messages were sent, run:")
    print("  docker exec -it dq_kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic data-quality-metrics --from-beginning")