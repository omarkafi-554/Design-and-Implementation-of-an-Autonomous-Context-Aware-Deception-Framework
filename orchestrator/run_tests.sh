#!/usr/bin/env bash
# run_tests.sh - run every orchestrator test file; exit code = number of failed files.
# Usage (from anywhere):  bash ~/deception-framework/orchestrator/run_tests.sh
cd "$(dirname "$0")/.." || exit 99            # project root: ~/deception-framework
log=$(mktemp)
fail=0
for t in orchestrator/tests/test_*.py; do
    printf '%-40s ' "$t"
    if python3 "$t" > "$log" 2>&1; then
        tail -n 1 "$log"
    else
        echo "FAILED"
        sed 's/^/    /' "$log"
        fail=$((fail + 1))
    fi
done
rm -f "$log"
echo
if [ "$fail" -eq 0 ]; then echo "[+] all test files passed"; else echo "[-] $fail test file(s) failed"; fi
exit "$fail"
