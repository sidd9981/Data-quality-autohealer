"""
Kafka Consumer for processing quality metrics
"""

from confluent_kafka import Consumer, KafkaError
import json
from typing import Dict, Callable
import sys
sys.path.append('.')

from src.detectors.ensemble_classifier import QualityEnsembleClassifier
from src.streaming.kafka_producer import QualityMetricsProducer


class QualityMetricsConsumer:
    """Consume quality metrics and trigger detection"""
    
    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9093',
        group_id: str = 'quality-detector-group'
    ):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        
        self.producer = QualityMetricsProducer(bootstrap_servers)
        self.ensemble = None
        
        self.topics = ['data-quality-metrics']
    
    def load_ensemble(self):
        """Load ensemble classifier"""
        print("Loading ensemble classifier...")
        self.ensemble = QualityEnsembleClassifier()
        self.ensemble.load_detectors()
        print("Ensemble loaded")
    
    def process_quality_metrics(self, message: Dict):
        """Process quality metrics and detect issues"""
        
        pipeline_id = message.get('pipeline_id')
        profile = message.get('metrics')
        timestamp = message.get('timestamp')
        
        if not profile:
            print(f"No metrics in message for pipeline {pipeline_id}")
            return
        
        # Detect quality issues
        detected_issues, all_scores = self.ensemble.predict_issue_types_multi(profile)
        
        print(f"\nPipeline: {pipeline_id}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Detected: {detected_issues}")
        print(f"  Scores: {', '.join([f'{k}={v:.2f}' for k, v in all_scores.items()])}")
        
        # If issues detected, send alert
        if 'clean' not in detected_issues:
            severity = self._determine_severity(all_scores)
            
            print(f"  Sending alert (severity: {severity})...")
            self.producer.send_quality_alert(
                pipeline_id=pipeline_id,
                issue_types=detected_issues,
                severity=severity
            )
            
            # Trigger remediation for each issue
            for issue_type in detected_issues:
                print(f"  Triggering remediation for {issue_type}...")
                self.producer.send_remediation_action(
                    pipeline_id=pipeline_id,
                    issue_type=issue_type,
                    action_type=self._get_remediation_action(issue_type),
                    status='triggered',
                    details={'confidence': float(all_scores[issue_type])}
                )
    
    def _determine_severity(self, scores: Dict[str, float]) -> str:
        """Determine alert severity based on scores"""
        max_score = max(scores.values())
        
        if max_score > 0.9:
            return 'critical'
        elif max_score > 0.8:
            return 'high'
        elif max_score > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _get_remediation_action(self, issue_type: str) -> str:
        """Map issue type to remediation action"""
        action_map = {
            'schema_drift': 're_ingest',
            'distribution_shift': 'retrain_model',
            'missing_data': 'impute',
            'outlier': 'quarantine',
            'type_mismatch': 'coerce_types'
        }
        return action_map.get(issue_type, 'alert')
    
    def start_consuming(self, callback: Callable = None):
        """Start consuming messages"""
        
        if self.ensemble is None:
            self.load_ensemble()
        
        self.consumer.subscribe(self.topics)
        
        print(f"Consuming from topics: {self.topics}")
        print("Press Ctrl+C to stop...\n")
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f'Error: {msg.error()}')
                        break
                
                # Parse message
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    
                    # Process with ensemble or custom callback
                    if callback:
                        callback(message)
                    else:
                        self.process_quality_metrics(message)
                        
                except json.JSONDecodeError as e:
                    print(f"Failed to parse message: {e}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            print("\nStopping consumer...")
        finally:
            self.consumer.close()
            self.producer.close()


if __name__ == "__main__":
    print("Starting Quality Metrics Consumer...")
    
    consumer = QualityMetricsConsumer()
    consumer.start_consuming()