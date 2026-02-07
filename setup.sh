#!/bin/bash

# Data Quality Auto-Healing System - Setup Script
# Run this to initialize the project

set -e

echo "Setting up Data Quality Auto-Healing System..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Create project directory structure
echo -e "${BLUE}Creating project directory structure...${NC}"

mkdir -p data/{raw,processed,training,validation}
mkdir -p models/{detectors,ensemble}
mkdir -p src/{detectors,profilers,remediation,streaming,api}
mkdir -p src/remediation/airflow_dags
mkdir -p configs/{quality_thresholds,remediation_policies,monitoring}
mkdir -p tests/{unit,integration}
mkdir -p notebooks
mkdir -p logs
mkdir -p docker
mkdir -p airflow/dags
mkdir -p dashboard/grafana_dashboards

echo -e "${GREEN}Directory structure created${NC}"

# 2. Create Python virtual environment
echo -e "${BLUE}Creating Python virtual environment...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Virtual environment created${NC}"

# 3. Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# 4. Install requirements
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Dependencies installed${NC}"

# 5. Create .env file
echo -e "${BLUE}Creating .env configuration...${NC}"

cat > .env << 'EOF'
# Database
DATABASE_URL=postgresql://dquser:dqpass@localhost:5432/data_quality
POSTGRES_USER=dquser
POSTGRES_PASSWORD=dqpass
POSTGRES_DB=data_quality

# Redis
REDIS_URL=redis://localhost:6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9093

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://dquser:dqpass@localhost:5432/airflow

# API
API_PORT=8000
API_HOST=0.0.0.0

# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin
EOF

echo -e "${GREEN}.env file created${NC}"

# 6. Create config files
echo -e "${BLUE}Creating configuration files...${NC}"

# Quality thresholds config
cat > configs/quality_thresholds.yaml << 'EOF'
# Quality threshold configurations

schema_drift:
  detection_threshold: 0.8
  alert_threshold: 0.9
  
distribution_shift:
  kl_divergence_threshold: 0.15
  wasserstein_threshold: 0.20
  alert_threshold: 0.25
  
missing_data:
  acceptable_rate: 0.05
  alert_rate: 0.15
  
outliers:
  iqr_multiplier: 3.0
  z_score_threshold: 3.5
  alert_threshold: 0.10

data_type_mismatch:
  tolerance: 0.02
  alert_threshold: 0.05
  
correlation_break:
  min_correlation_change: 0.3
  alert_threshold: 0.5
EOF

# Remediation policies config
cat > configs/remediation_policies.yaml << 'EOF'
# Auto-remediation policies

schema_drift:
  action: re_ingest
  retry_attempts: 3
  escalate_after: 3
  
distribution_shift:
  action: retrain_model
  min_samples_required: 1000
  escalate_after: 2
  
missing_data:
  action: impute
  methods: [mean, median, knn]
  fallback: drop_rows
  
outliers:
  action: quarantine
  review_required: true
  statistical_correction: true
  
data_type_mismatch:
  action: coerce_types
  fallback: reject_batch
  
correlation_break:
  action: alert_data_scientist
  auto_fix: false
EOF

# Prometheus config
cat > configs/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quality_api'
    static_configs:
      - targets: ['quality_api:8000']
  
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']
EOF

echo -e "${GREEN}Configuration files created${NC}"

# 7. Create Dockerfiles
echo -e "${BLUE}Creating Dockerfiles...${NC}"

# API Dockerfile
cat > docker/Dockerfile.api << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "src.api.quality_service:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Airflow Dockerfile
cat > docker/Dockerfile.airflow << 'EOF'
FROM apache/airflow:2.7.3-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root configs/ /opt/airflow/configs/
EOF

echo -e "${GREEN}Dockerfiles created${NC}"

# 8. Create initial Python files
echo -e "${BLUE}Creating initial Python modules...${NC}"

# __init__ files
touch src/__init__.py
touch src/detectors/__init__.py
touch src/profilers/__init__.py
touch src/remediation/__init__.py
touch src/streaming/__init__.py
touch src/api/__init__.py
touch tests/__init__.py

# Create a simple test file
cat > tests/test_setup.py << 'EOF'
"""Test to verify setup is correct"""

def test_imports():
    """Test that all dependencies can be imported"""
    import torch
    import pandas as pd
    import numpy as np
    from pyspark.sql import SparkSession
    import kafka
    assert True
EOF

echo -e "${GREEN}Python modules initialized${NC}"

# 9. Create .gitignore
echo -e "${BLUE}Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Data
data/raw/*
data/processed/*
!data/.gitkeep

# Models
models/*.pth
models/*.pkl
!models/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Jupyter
.ipynb_checkpoints/

# Docker
docker-compose.override.yml

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/

# Airflow
airflow.db
airflow.cfg
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary
*.tmp
*.bak
EOF

echo -e "${GREEN}.gitignore created${NC}"

# 10. Create README
echo -e "${BLUE}Creating README...${NC}"

cat > README.md << 'EOF'
# Data Quality Auto-Healing System

An intelligent data quality monitoring system that autonomously detects, diagnoses, and remediates data pipeline failures in real-time.

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   ```

2. **Start infrastructure:**
   ```bash
   docker-compose up -d
   ```

3. **Verify services:**
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)
   - Airflow: http://localhost:8080 (admin/admin)
   - API: http://localhost:8000/docs
   - Prometheus: http://localhost:9090

4. **Run tests:**
   ```bash
   source venv/bin/activate
   pytest tests/
   ```

## Project Structure

```
data-quality-autohealer/
├── src/              # Source code
├── data/             # Data files
├── models/           # Trained models
├── configs/          # Configuration files
├── tests/            # Test files
├── notebooks/        # Jupyter notebooks
├── airflow/          # Airflow DAGs
├── docker/           # Docker files
└── dashboard/        # Grafana dashboards
```

## Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run quality API locally
uvicorn src.api.quality_service:app --reload

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## Next Steps

See `notebooks/01_getting_started.ipynb` for a walkthrough.
EOF

echo -e "${GREEN}README created${NC}"

echo ""
echo -e "${GREEN}Setup complete${NC}"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Docker services: docker-compose up -d"
echo "3. Check service status: docker-compose ps"
echo "4. Access services:"
echo "   - MLflow: http://localhost:5000"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo "   - Airflow: http://localhost:8080 (admin/admin)"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Ready to start building"