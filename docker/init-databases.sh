#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO dquser;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO dquser;
EOSQL

echo "Additional databases created: airflow, mlflow"