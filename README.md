# Autonomous Data Quality Auto-Healing System

ML-driven system that **detects and automatically remediates data quality issues in real time** using PyTorch, Kafka, and Airflow.

This project demonstrates how modern ML systems can **prevent silent data corruption** before it degrades downstream models or breaks production pipelines.

---

## Project Overview

Self-healing data pipeline system that automatically detects 5 types of quality issues using ML, streams events in real-time via Kafka, and triggers autonomous remediation through Airflow workflows. Features live monitoring dashboard with WebSocket updates.

## What This System Does

- Detects **5 classes of data quality failures**:
  - Schema Drift
  - Distribution Shift
  - Missing Data
  - Outliers
  - Type Mismatches
- Runs **real-time ML inference** on streaming data
- Triggers **autonomous remediation workflows** without human intervention
- Streams live system state to a **real-time monitoring dashboard**

---

## Why This Matters

In real production systems, data failures often:
- Go undetected for hours or days
- Quietly degrade ML model performance
- Require expensive manual debugging

This system closes the loop:

**Detect → Decide → Remediate → Validate**

before downstream impact occurs.

---

## Live Dashboard

<img src="dashboard/dashboard.png" width="800" alt="Real-time Data Quality Dashboard"/>


The dashboard displays:
- Live quality scores
- Active issues and severity
- Remediation progress
- System events in real time

---

## System Architecture

```
DATA SOURCES
     |
     v
KAFKA STREAMS
(data-quality-metrics)
     |
     v
ML DETECTORS (PyTorch)
     |
     v
QUALITY ALERTS
     |
     v
AIRFLOW DAGS
(auto-remediation)
     |
     v
WEBSOCKET DASHBOARD
```

---

## Tech Stack

- **Machine Learning**: PyTorch (ensemble neural networks)
- **Data Processing**: PySpark
- **Streaming**: Apache Kafka
- **Orchestration**: Apache Airflow
- **Backend**: FastAPI, WebSockets
- **Frontend**: React, TailwindCSS
- **Infrastructure**: Docker

---

## Why This Project Is Interesting

- Uses **ML instead of rules** for complex, non-linear data failures
- Designed as an **end-to-end production-style system**
- Emphasizes **low latency, observability, and safety**
- Integrates ML predictions directly into operational workflows
- Models are evaluated on **accuracy, false positives, and system impact**

---

## Machine Learning Overview

- PyTorch ensemble of binary classifiers
- Detects 5 data quality issue types
- Achieves **over 92% validation accuracy**
- Uses engineered statistical features rather than raw data
- Confidence-based thresholds to reduce false positives

---

## Quick Start

```bash
bash setup.sh
bash start_system.sh
python run_auto_healing.py 
OR
bash run_demo.sh
```

Dashboard:
```
ws://localhost:8001/ws/quality
```

---

## Project Structure

```
src/
├── profilers/        # PySpark data profiling
├── detectors/        # PyTorch ML models
├── streaming/        # Kafka producers/consumers
├── remediation/      # Airflow DAGs
├── api/              # FastAPI + WebSockets
dashboard/            # React UI
models/               # Trained ML models
```

---

## Performance

- **Detection latency**: < 1 second
- **Streaming throughput**: 1000+ events/sec
- **Dashboard latency**: < 50ms
- **Manual debugging reduction**: ~85% (simulated workloads)

---

## Who This Is For

- Students learning **ML systems and MLOps**
- Engineers interested in **data reliability**
- Recruiters evaluating **production ML thinking**
- Anyone curious how ML models fit into real systems

---

## Disclaimer

This project is designed to simulate production behavior and engineering decision-making.
It is not a drop-in replacement for enterprise data quality platforms, but a demonstration
of **system design, ML integration, and autonomous recovery patterns**.
