#!/bin/bash
# Monitor retraining and benchmarking progress

echo "Monitoring follow-up processes..."
echo "Retraining PID: 42820"
echo "Benchmarking PID: 42821"
echo ""

check_progress() {
    echo "=== $(date) ==="
    
    # Check retraining
    if ps aux | grep -q "42820"; then
        echo "✓ Retraining still running"
        RETRAIN_SIZE=$(wc -l < data/experiments/retrain_production.log 2>/dev/null || echo 0)
        echo "  Log size: $RETRAIN_SIZE lines"
        tail -3 data/experiments/retrain_production.log | grep -v "^$" | tail -1
    else
        echo "✗ Retraining completed"
        if [ -f data/experiments/retrain_production.log ]; then
            grep -i "complete\|saved\|error" data/experiments/retrain_production.log | tail -2
        fi
    fi
    
    # Check benchmarking
    if ps aux | grep -q "42821"; then
        echo "✓ Benchmarking still running"
        BENCH_SIZE=$(wc -l < data/experiments/benchmark_production.log 2>/dev/null || echo 0)
        echo "  Log size: $BENCH_SIZE lines"
        tail -3 data/experiments/benchmark_production.log | grep -v "^$" | tail -1
    else
        echo "✗ Benchmarking completed"
        if [ -f data/experiments/benchmark_production.log ]; then
            grep -i "complete\|output\|error" data/experiments/benchmark_production.log | tail -2
        fi
    fi
    
    echo ""
}

# Initial check
check_progress

# Monitor for up to 6 hours
for i in {1..60}; do
    echo "Waiting 6 minutes... (iteration $i/60)"
    sleep 360
    check_progress
done

echo "Monitoring timeout - processes still running or completed"
