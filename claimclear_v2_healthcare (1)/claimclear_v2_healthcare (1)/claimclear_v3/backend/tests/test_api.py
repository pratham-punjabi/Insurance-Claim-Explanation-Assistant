"""
Basic API tests — run with: pytest tests/test_api.py -v
Requires backend running at http://localhost:8000
"""
import pytest
import httpx

BASE = "http://localhost:8000"

def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_list_claims():
    r = httpx.get(f"{BASE}/api/claims")
    assert r.status_code == 200
    data = r.json()
    assert "claims" in data
    assert data["total"] >= 7

def test_filter_claims_denied():
    r = httpx.get(f"{BASE}/api/claims?status=denied")
    assert r.status_code == 200
    claims = r.json()["claims"]
    assert all(c["status"] == "denied" for c in claims)

def test_get_claim():
    r = httpx.get(f"{BASE}/api/claims/CLM-2024-0041")
    assert r.status_code == 200
    c = r.json()
    assert c["id"] == "CLM-2024-0041"
    assert c["status"] == "denied"

def test_list_policies():
    r = httpx.get(f"{BASE}/api/policies")
    assert r.status_code == 200
    data = r.json()
    assert len(data["policies"]) == 5

def test_index_stats():
    r = httpx.get(f"{BASE}/api/index/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_chunks" in data

def test_dashboard_stats():
    r = httpx.get(f"{BASE}/api/claims/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_claims"] >= 7
    assert "approved" in data
