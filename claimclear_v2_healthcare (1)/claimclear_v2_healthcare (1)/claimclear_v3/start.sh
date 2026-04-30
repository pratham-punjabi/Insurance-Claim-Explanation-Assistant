#!/bin/bash
echo "🏥 ClaimClear v2 — Healthcare Insurance Assistant"
echo "================================================="
cd "$(dirname "$0")/backend"
pip install -r requirements.txt -q
uvicorn main:app --reload --port 8000 --host 0.0.0.0
