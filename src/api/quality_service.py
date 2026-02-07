"""
FastAPI service for data quality checks
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import io
import sys
sys.path.append('.')

from src.profilers.spark_profiler import SparkDataProfiler
from src.detectors.ensemble_classifier import QualityEnsembleClassifier
from src.streaming.kafka_producer import QualityMetricsProducer

app = FastAPI(title="Data Quality API", version="1.0.0")

# Global instances
profiler = SparkDataProfiler()
ensemble = QualityEnsembleClassifier()
producer = QualityMetricsProducer()

# Load ensemble on startup
@app.on_event("startup")
async def startup_event():
    print("Loading ensemble classifier...")
    ensemble.load_detectors()
    print("API ready")


class QualityCheckRequest(BaseModel):
    pipeline_id: str
    data_csv: str
    baseline_csv: Optional[str] = None


class QualityCheckResponse(BaseModel):
    pipeline_id: str
    detected_issues: List[str]
    scores: Dict[str, float]
    severity: str
    recommendations: List[str]


@app.get("/")
async def root():
    return {"message": "Data Quality API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "detectors_loaded": len(ensemble.detectors)}


@app.post("/quality/check", response_model=QualityCheckResponse)
async def check_quality(file: UploadFile = File(...), baseline_file: UploadFile = File(None)):
    """
    Check data quality for uploaded CSV file
    """
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Read baseline if provided
        baseline_df = None
        if baseline_file:
            baseline_contents = await baseline_file.read()
            baseline_df = pd.read_csv(io.BytesIO(baseline_contents))
        
        # Profile data
        profile = profiler.profile_dataset(df, baseline_df=baseline_df)
        
        # Detect issues
        detected_issues, all_scores = ensemble.predict_issue_types_multi(profile)
        
        # Determine severity
        max_score = max(all_scores.values()) if all_scores else 0.0
        if max_score > 0.9:
            severity = 'critical'
        elif max_score > 0.8:
            severity = 'high'
        elif max_score > 0.6:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate recommendations
        recommendations = []
        for issue in detected_issues:
            if issue == 'clean':
                recommendations.append("No quality issues detected")
            elif issue == 'schema_drift':
                recommendations.append("Re-ingest data with updated schema")
            elif issue == 'distribution_shift':
                recommendations.append("Retrain models on new data distribution")
            elif issue == 'missing_data':
                recommendations.append("Impute missing values or investigate data pipeline")
            elif issue == 'outlier':
                recommendations.append("Quarantine outliers and review data collection")
            elif issue == 'type_mismatch':
                recommendations.append("Fix data type coercion in pipeline")
        
        # Send to Kafka
        pipeline_id = file.filename.replace('.csv', '')
        producer.send_quality_metrics(pipeline_id, profile)

        if 'clean' not in detected_issues:
            producer.send_quality_alert(
                pipeline_id=pipeline_id,
                issue_types=detected_issues,
                severity=severity
            )
        
        return QualityCheckResponse(
            pipeline_id=pipeline_id,
            detected_issues=detected_issues,
            scores={k: float(v) for k, v in all_scores.items()},
            severity=severity,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quality/status/{pipeline_id}")
async def get_status(pipeline_id: str):
    """Get quality status for a pipeline"""
    return {
        "pipeline_id": pipeline_id,
        "status": "monitoring",
        "message": "Quality checks running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)