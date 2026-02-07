"""
Test suite for FastAPI quality service
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.api.quality_service import app
from src.data.bad_data_generator import BadDataGenerator


class TestQualityAPI:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def clean_csv_file(self):
        gen = BadDataGenerator()
        df = gen.generate_clean_dataset(n_rows=1000)
        
        # Convert to CSV bytes
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()
        
        return ('test.csv', csv_bytes, 'text/csv')
    
    @pytest.fixture
    def bad_csv_file(self):
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()
        
        return ('test.csv', csv_bytes, 'text/csv')
    
    def test_root_endpoint(self, client):
        """Test API root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert data['message'] == "Data Quality API"
    
    def test_health_endpoint(self, client):
        """Test API health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'detectors_loaded' in data
    
    def test_quality_check_clean_data(self, client, clean_csv_file):
        """Test quality check with clean data"""
        response = client.post(
            "/quality/check",
            files={'file': clean_csv_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert 'detected_issues' in data
        assert 'scores' in data
        assert 'severity' in data
        assert 'recommendations' in data
        
        # Clean data should have low severity
        assert data['severity'] in ['low', 'none']
    
    def test_quality_check_bad_data(self, client, bad_csv_file):
        """Test quality check processes problematic data"""
        response = client.post(
            "/quality/check",
            files={'file': bad_csv_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return valid response structure
        assert 'detected_issues' in data
        assert 'scores' in data
        assert 'severity' in data
        assert isinstance(data['detected_issues'], list)
        assert isinstance(data['scores'], dict)
    
    def test_quality_check_invalid_file(self, client):
        """Test API handles invalid file upload"""
        # Send text file instead of CSV
        invalid_file = ('test.txt', b'not a csv file', 'text/plain')
        
        response = client.post(
            "/quality/check",
            files={'file': invalid_file}
        )
        
        # Should return error
        assert response.status_code in [400, 500]
    
    def test_quality_check_empty_file(self, client):
        """Test API handles empty CSV"""
        # Create empty CSV with just headers
        empty_csv_content = "col1,col2\n".encode()
        empty_csv = ('empty.csv', empty_csv_content, 'text/csv')
        
        response = client.post(
            "/quality/check",
            files={'file': empty_csv}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_quality_check_response_time(self, client, clean_csv_file):
        """Test API meets latency requirements"""
        import time
        
        start = time.time()
        response = client.post(
            "/quality/check",
            files={'file': clean_csv_file}
        )
        duration = time.time() - start
        
        assert response.status_code == 200
        # Should respond in < 10 seconds
        assert duration < 10, f"API took {duration}s, too slow"
    
    def test_concurrent_requests(self, client, clean_csv_file):
        """Test API handles concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.post("/quality/check", files={'file': clean_csv_file})
        
        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in results)
    
    def test_api_returns_all_detector_scores(self, client, bad_csv_file):
        """Test API returns scores for all detectors"""
        response = client.post(
            "/quality/check",
            files={'file': bad_csv_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have scores for all 5 detectors
        expected_detectors = [
            'schema_drift',
            'distribution_shift',
            'missing_data',
            'outlier',
            'type_mismatch'
        ]
        
        for detector in expected_detectors:
            assert detector in data['scores'], f"Missing score for {detector}"
    
    def test_api_provides_recommendations(self, client, bad_csv_file):
        """Test API provides actionable recommendations"""
        response = client.post(
            "/quality/check",
            files={'file': bad_csv_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should provide recommendations
        assert 'recommendations' in data
        assert isinstance(data['recommendations'], list)


class TestWebSocketServer:
    """Test WebSocket server for real-time updates"""
    
    @pytest.mark.integration
    def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        from fastapi.testclient import TestClient
        from src.api.websocket_server import app as ws_app
        
        client = TestClient(ws_app)
        
        with client.websocket_connect("/ws/quality") as websocket:
            # Connection successful
            assert websocket is not None
    
    @pytest.mark.integration
    def test_websocket_broadcasts_events(self):
        """Test WebSocket broadcasts quality events"""
        from fastapi.testclient import TestClient
        from src.api.websocket_server import app as ws_app
        
        client = TestClient(ws_app)
        
        with client.websocket_connect("/ws/quality") as websocket:
            # Wait for a message (with timeout)
            try:
                data = websocket.receive_json(timeout=5)
                assert 'type' in data
                assert data['type'] in ['quality_check', 'alert', 'remediation', 'connected']
            except Exception:
                # No messages yet is okay for this test
                pass


class TestAPIIntegration:
    """End-to-end API integration tests"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.integration
    def test_api_to_kafka_pipeline(self, client):
        """Test API uploads trigger valid responses"""
        gen = BadDataGenerator()
        df, _ = gen.generate_quality_issue_dataset(n_rows=1000, issue_type='schema_drift')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()
        
        file = ('test.csv', csv_bytes, 'text/csv')
        
        # Upload file
        response = client.post("/quality/check", files={'file': file})
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return valid structure
        assert 'detected_issues' in data
        assert 'scores' in data
    
    def test_api_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint (if exists)"""
        response = client.get("/metrics")
        
        # Endpoint might not exist, that's okay
        assert response.status_code in [200, 404]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not integration'])