#!/bin/bash

# Complete Demo Script - Runs full auto-remediation demo

echo "Data Quality Auto-Healing DEMO"

# Check if system is running
if ! pgrep -f "websocket_server.py" > /dev/null; then
    echo "Error: System not running. Start with ./start_system.sh first"
    exit 1
fi

echo ""
echo "Running complete auto-remediation demonstration..."
echo "Watch your dashboard at: http://localhost:8001/dashboard/index.html"
echo ""
echo "Press ENTER to start demo..."
read

# Run the auto-healing demo
python run_auto_healing.py

echo ""
echo "Demo Complete!"
echo ""
echo "Check your dashboard to see:"
echo "  - Quality checks performed"
echo "  - Issues detected and remediated"
echo "  - Real-time metrics updated"
echo ""