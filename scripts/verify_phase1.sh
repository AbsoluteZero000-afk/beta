#!/usr/bin/env bash
# ============================================
# QuantEdge — Phase 1 Verification Script
# Run: bash scripts/verify_phase1.sh
# ============================================

set -e

echo "========================================"
echo "QuantEdge Phase 1 — Verification"
echo "========================================"

echo ""
echo "[1/8] Checking Docker..."
docker --version || { echo "FAIL: Docker not installed"; exit 1; }
echo "  OK"

echo ""
echo "[2/8] Checking Docker Compose..."
docker compose version || { echo "FAIL: Docker Compose not found"; exit 1; }
echo "  OK"

echo ""
echo "[3/8] Checking Python 3.12..."
python3 --version | grep "3.12" || { echo "FAIL: Python 3.12 not found"; exit 1; }
echo "  OK"

echo ""
echo "[4/8] Verifying pip packages..."
python3 -c "import asyncpg; print(f'  asyncpg {asyncpg.__version__}')"
python3 -c "import redis; print(f'  redis-py {redis.__version__}')"
python3 -c "import redis.asyncio; print('  redis.asyncio OK')"
python3 -c "import uvloop; print(f'  uvloop {uvloop.__version__}')"
python3 -c "import numba; print(f'  numba {numba.__version__}')"
python3 -c "import pandas; print(f'  pandas {pandas.__version__}')"
python3 -c "import orjson; print(f'  orjson OK')"
python3 -c "import pydantic; print(f'  pydantic {pydantic.__version__}')"
python3 -c "import alpaca; print('  alpaca-py OK')"
echo "  All packages OK"

echo ""
echo "[5/8] Checking Docker containers..."
docker compose up -d --wait
echo "  Containers started"

echo ""
echo "[6/8] Verifying TimescaleDB..."
docker exec quantedge-timescaledb psql -U quantedge -d quantedge -c "SELECT default_version FROM pg_available_extensions WHERE name='timescaledb';" || { echo "FAIL: TimescaleDB check failed"; exit 1; }
echo "  TimescaleDB OK"

echo ""
echo "[7/8] Verifying Redis..."
docker exec quantedge-redis redis-cli ping || { echo "FAIL: Redis check failed"; exit 1; }
echo "  Redis OK"

echo ""
echo "[8/8] Running Phase 1 app check..."
# Set env vars for local testing (pointing to localhost)
export POSTGRES_HOST=localhost
export REDIS_HOST=localhost
python3 -m src.main
echo ""
echo "========================================"
echo "Phase 1 — ALL CHECKS PASSED ✅"
echo "========================================"
