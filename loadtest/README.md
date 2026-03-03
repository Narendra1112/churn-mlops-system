# Churn Inference API – Load Testing

This directory contains Locust-based load testing for the `/predict` endpoint.

---

## Target

Default target:
http://127.0.0.1:8000

Make sure the API is running before executing tests.

```bash
uvicorn api.app:app --reload
```

---

## Requirements

Install Locust inside your virtual environment:

```bash
pip install locust
```

---

## Payload

The test uses `payload.json` from the project root.

If needed, override:

```bash
set PAYLOAD_PATH=path\to\custom_payload.json
```

---

## Test Scenarios

### 1️⃣ Baseline Run (Light Load)

```bash
locust -f loadtest/locustfile.py --headless -u 25 -r 5 -t 60s --csv loadtest/results/run_u25
```

- Users: 25
- Spawn rate: 5/sec
- Duration: 60 seconds

---

### 2️⃣ Target Load

```bash
locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s --csv loadtest/results/run_u50
```

- Users: 50
- Spawn rate: 10/sec
- Duration: 60 seconds

---

### 3️⃣ Stress Test

```bash
locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 60s --csv loadtest/results/run_u100
```

- Users: 100
- Spawn rate: 20/sec
- Duration: 60 seconds

---

## Output Files

Each run generates:

- `*_stats.csv`
- `*_failures.csv`
- `*_stats_history.csv`

Location:
```
loadtest/results/
```

---

## What To Evaluate

From the CSV output, extract:

- Requests per second (RPS)
- p50 latency
- p95 latency
- p99 latency
- Failure count
- 429 (Backpressure) rate
- 504 (Timeout) rate

---

## Expected Behavior

- Under baseline load → 0 failures
- Under stress → 429 responses if concurrency limit is reached
- `/metrics` endpoint should remain responsive
- No crash of uvicorn process
- No unbounded latency growth

---

## Example Performance Targets

For a local dev machine:

- p50 < 300ms (baseline)
- p95 < 600ms (target load)
- Controlled 429 under stress
- No unexpected 500 errors

---

## Notes

- Ensure `MAX_INFLIGHT_INFERENCES` is configured correctly.
- Ensure `INFERENCE_QUEUE_TIMEOUT_S` is set to a small value (e.g., 0.05).
- Restart API after config changes.
- Clear background PowerShell jobs before new runs:

```bash
Get-Job | Remove-Job -Force
```

---

This load test validates:

- Concurrency control
- Backpressure behavior
- Timeout handling
- Stability under stress
- Metrics exposure
