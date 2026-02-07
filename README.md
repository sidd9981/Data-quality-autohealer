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
