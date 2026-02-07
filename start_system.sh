#!/bin/bash

# Data Quality Auto-Healing System - Complete Startup Script
# Starts all services: Kafka, Airflow, WebSocket Server, Dashboard

set -e

echo "Starting Data Quality Auto-Healing System"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

source venv/bin/activate

# Start Docker services (Kafka, Zookeeper, Airflow)
echo ""
echo "1. Starting Docker services..."
docker-compose up -d

echo "   Waiting for services to be ready..."
sleep 10

# Check if services are running
echo ""
echo "2. Checking service health..."
docker-compose ps

# Start WebSocket server in background
echo ""
echo "3. Starting WebSocket server..."
python src/api/websocket_server.py > logs/websocket.log 2>&1 &
WEBSOCKET_PID=$!
echo "   WebSocket server started (PID: $WEBSOCKET_PID)"

# Wait for WebSocket to be ready
sleep 3

# Run initial demo to populate dashboard
echo ""
echo "4. Running initial demo to populate dashboard..."
python run_auto_healing.py > logs/demo.log 2>&1 &
DEMO_PID=$!
echo "   Demo running (PID: $DEMO_PID)"

# Wait a bit for demo to generate some events
sleep 5

# Open dashboard in browser
echo ""
echo "5. Opening dashboard..."
DASHBOARD_PATH="$(pwd)/dashboard/index.html"
if command -v open &> /dev/null; then
    # macOS
    open "file://${DASHBOARD_PATH}"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "file://${DASHBOARD_PATH}"
fi

echo ""
echo "System Started Successfully!"
echo ""
echo "Services running:"
echo "  - Dashboard: file://$(pwd)/dashboard/index.html (opened in browser)"
echo "  - WebSocket: ws://localhost:8001/ws/quality"
echo "  - Kafka: localhost:9092"
echo "  - Airflow: http://localhost:8080 (user: airflow, pass: airflow)"
echo ""
echo "To test the system:"
echo "  python test_real_remediation.py"
echo ""
echo "To stop all services:"
echo "  ./stop_system.sh"
echo ""
echo "WebSocket PID: $WEBSOCKET_PID (saved to .websocket.pid)"
echo $WEBSOCKET_PID > .websocket.pid
echo ""