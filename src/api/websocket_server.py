"""
WebSocket server for streaming quality updates to dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import List, Dict
from datetime import datetime
from confluent_kafka import Consumer, KafkaError
import threading
import sys
sys.path.append('.')

app = FastAPI(title="Quality WebSocket Server")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# Kafka consumer for quality events
class QualityEventsConsumer:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': 'localhost:9093',
            'group.id': 'dashboard-consumer',
            'auto.offset.reset': 'latest'
        })
        
        self.consumer.subscribe([
            'data-quality-metrics',
            'quality-alerts',
            'remediation-actions'
        ])
        
        self.running = False
    
    def consume_loop(self):
        """Consume Kafka messages and broadcast to WebSocket clients"""
        self.running = True
        
        print("Kafka consumer started, listening for quality events...")
        
        while self.running:
            msg = self.consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Kafka error: {msg.error()}")
                continue
            
            try:
                value = json.loads(msg.value().decode('utf-8'))
                topic = msg.topic()
                
                # Create event for dashboard
                event = {
                    'type': self._get_event_type(topic),
                    'topic': topic,
                    'data': value,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Broadcast to all connected clients
                asyncio.run(manager.broadcast(event))
                
                print(f"Broadcasted event: {event['type']}")
                
            except Exception as e:
                print(f"Error processing message: {e}")
    
    def _get_event_type(self, topic: str) -> str:
        """Map Kafka topic to event type"""
        if topic == 'data-quality-metrics':
            return 'quality_check'
        elif topic == 'quality-alerts':
            return 'alert'
        elif topic == 'remediation-actions':
            return 'remediation'
        return 'unknown'
    
    def start(self):
        """Start consumer in background thread"""
        thread = threading.Thread(target=self.consume_loop, daemon=True)
        thread.start()
        print("Kafka consumer thread started")
    
    def stop(self):
        """Stop consumer"""
        self.running = False
        self.consumer.close()

kafka_consumer = QualityEventsConsumer()


@app.on_event("startup")
async def startup():
    """Start Kafka consumer on app startup"""
    kafka_consumer.start()
    print("WebSocket server ready")


@app.on_event("shutdown")
async def shutdown():
    """Stop Kafka consumer on shutdown"""
    kafka_consumer.stop()


@app.websocket("/ws/quality")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for quality updates"""
    
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to quality monitoring',
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            
            if data == 'ping':
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "kafka_consumer": "running" if kafka_consumer.running else "stopped"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)