#!/bin/bash

# Data Quality Auto-Healing System - Shutdown Script

echo "Stopping Data Quality Auto-Healing System"

# Stop WebSocket server
echo ""
echo "1. Stopping WebSocket server..."
pkill -f "python src/api/websocket_server.py" || echo "   WebSocket server not running"

# Stop Docker services
echo ""
echo "2. Stopping Docker services..."
docker-compose down

echo ""
echo "System Stopped"