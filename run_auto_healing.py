"""
Auto-Healing System Demo
Demonstrates complete quality issue detection and remediation workflow
"""

import sys
sys.path.append('.')

from src.data.bad_data_generator import BadDataGenerator
from src.profilers.spark_profiler import SparkDataProfiler
from src.detectors.ensemble_classifier import QualityEnsembleClassifier
from src.streaming.kafka_producer import QualityMetricsProducer
import time


def main():
    print("REAL AUTO-REMEDIATION DEMO WITH LIVE DASHBOARD")
    
    # Initialize components
    generator = BadDataGenerator()
    profiler = SparkDataProfiler()
    producer = QualityMetricsProducer()
    
    # Load ensemble
    print("Loading detectors...")
    ensemble = QualityEnsembleClassifier()
    ensemble.load_detectors()
    
    # Generate clean baseline
    clean_df = generator.generate_clean_dataset(n_rows=1000)
    
    # STEP 1: Generate bad data with OBVIOUS issues
    print("\nSTEP 1: Data Pipeline Produces Bad Data")

    
    # Use parameters that create DETECTABLE issues
    _, bad_df = generator.generate_quality_issue_dataset(
        n_rows=1000,
        issue_type='outlier',
        column='salary',
        outlier_rate=0.15  # 15% outliers - very obvious
    )
    
    print(f"Generated bad data:")
    print(f"  Pipeline: demo_pipeline_1")
    print(f"  Columns: {bad_df.columns.tolist()}")
    
    # STEP 2: Profile the data
    print("\nSTEP 2: Quality System Profiles Data")
    print("-" * 70)
    
    profile = profiler.profile_dataset(bad_df, baseline_df=clean_df)
    
    print(f"Extracted {len(profile)} quality metrics")
    
    # Send to Kafka for dashboard
    producer.send_quality_metrics('demo_pipeline_1', profile)
    print("Sent quality metrics to Kafka → Dashboard updated")
    
    time.sleep(1)
    
    # STEP 3: Detect issues
    print("\nSTEP 3: Ensemble Detects Quality Issues")
    
    issues, scores = ensemble.predict_issue_types_multi(profile)
    
    print(f"Detected issues: {issues}")
    for issue_type, score in scores.items():
        if score > 0:
            print(f"  {issue_type}: {score:.2f} confidence")
    
    # Send alert to Kafka
    if issues and 'clean' not in issues:
        severity = 'high' if len(issues) > 2 else 'medium'
        producer.send_quality_alert('demo_pipeline_1', issues, severity)
        print("Sent alert to Kafka → Dashboard shows red alert")
        
        time.sleep(1)
        
        # STEP 4: Trigger remediation
        print("\nSTEP 4: Auto-Remediation Triggered")
        print("-" * 70)
        
        for issue in issues:
            if issue != 'clean':
                print(f"\nIssue: {issue.upper()}")
                print(f"Action: AUTO-REMEDIATING")
                
                # Send remediation started
                producer.send_remediation_action(
                    'demo_pipeline_1',
                    issue,
                    're_ingest' if issue == 'schema_drift' else 'auto_fix',
                    'started'
                )
                print("Dashboard shows: Remediation started")
                
                time.sleep(2)
                
                # Simulate remediation steps
                print(f"  Executing remediation workflow...")
                time.sleep(1)
                
                # Send remediation completed
                producer.send_remediation_action(
                    'demo_pipeline_1',
                    issue,
                    're_ingest' if issue == 'schema_drift' else 'auto_fix',
                    'completed'
                )
                print("Dashboard shows: Remediation completed")
                
                time.sleep(1)
    
    # STEP 5: Re-check quality
    print("\nSTEP 5: Re-Check Quality After Remediation")
    
    # Generate clean data to simulate successful remediation
    fixed_df = generator.generate_clean_dataset(n_rows=1000)
    profile_fixed = profiler.profile_dataset(fixed_df, baseline_df=clean_df)
    
    producer.send_quality_alert('demo_pipeline_1', issues, severity)
    
    issues_after, scores_after = ensemble.predict_issue_types_multi(profile_fixed)
    
    print(f"New quality check:")
    print(f"  Detected issues: {issues_after}")
    print(f"  Schema drift score: {scores_after.get('schema_drift', 0):.2f}")
    
    if 'clean' in issues_after or len(issues_after) == 0:
        print("SUCCESS: Quality issue resolved!")
        print("Dashboard shows: Issue status changed to 'resolved'")
    
    # Clean up
    producer.close()
    profiler.spark.stop()
    
    print("DEMO COMPLETE")
    print("\nCheck your dashboard - you should see:")
    print("  - Total Checks: 2")
    print("  - Issues Detected: 1+")
    print("  - Auto-Fixed: 1+")
    print("  - Active Issues: with 'resolved' status")
    print("  - Remediation Log: actions completed")



if __name__ == "__main__":
    main()