#!/bin/bash
# Quick GPU partition availability check for MSI.
# Shows, for each GPU partition:
#   - total nodes
#   - currently idle nodes (immediately available)
#   - allocated/mixed nodes (partially in use)
#   - pending jobs in the queue (competition)
#   - estimated wait for a new 1-GPU job
#
# Usage:
#   bash scripts/check_gpus.sh
#   watch -n 30 bash scripts/check_gpus.sh    # refresh every 30s

set -uo pipefail

GPU_PARTITIONS=(a100-4 a100-8 v100 interactive-gpu preempt-gpu)

printf "%-18s %6s %6s %6s %6s %8s %s\n" \
    "PARTITION" "TOTAL" "IDLE" "MIX" "ALLOC" "PENDING" "GPU_TYPE"
printf "%-18s %6s %6s %6s %6s %8s %s\n" \
    "---------" "-----" "----" "---" "-----" "-------" "--------"

for p in "${GPU_PARTITIONS[@]}"; do
    # Node counts by state
    idle=$(sinfo -h -p "$p" -t idle -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    mix=$(sinfo -h -p "$p"  -t mix  -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    alloc=$(sinfo -h -p "$p" -t alloc -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')
    total=$(sinfo -h -p "$p" -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')

    # If partition doesn't exist, skip
    if [ "$total" -eq 0 ]; then
        continue
    fi

    # Pending jobs in this partition
    pending=$(squeue -h -p "$p" -t PD 2>/dev/null | wc -l)

    # GPU type (from GRES string)
    gpu_type=$(sinfo -h -p "$p" -o "%G" 2>/dev/null | head -1 | \
               sed -E 's/.*gpu:([a-z0-9]+):.*/\1/' | head -1)

    printf "%-18s %6d %6d %6d %6d %8d %s\n" \
        "$p" "$total" "$idle" "$mix" "$alloc" "$pending" "$gpu_type"
done

echo ""
echo "Legend:"
echo "  IDLE  = fully free nodes, job starts immediately"
echo "  MIX   = partial use, 1-GPU jobs may fit into spare GRES"
echo "  ALLOC = fully allocated"
echo "  PEND  = other users waiting in queue for this partition"
echo ""

# Your own jobs
echo "Your jobs:"
my_jobs=$(squeue -h -u "$USER" 2>/dev/null)
if [ -z "$my_jobs" ]; then
    echo "  (none)"
else
    echo ""
    squeue -u "$USER" -o "%.10i %.12P %.20j %.8T %.10M %.10l %.6D %R"
fi

echo ""
echo "Tips:"
echo "  - If a100-4 shows 0 IDLE and high PEND: try v100 instead"
echo "  - MIX nodes can still accept jobs if they have spare GPUs"
echo "  - preempt-gpu usually has the shortest wait but jobs can be killed"
echo "  - interactive-gpu queue is separate from batch; short time limit"