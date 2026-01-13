#!/usr/bin/env bash
set -euo pipefail

(uvicorn backend.main:app --host 0.0.0.0 --port 8000 &)
BACKEND_PID=$!

cd frontend
npm install
npm run dev

kill ${BACKEND_PID}
