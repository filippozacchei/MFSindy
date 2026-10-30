#!/usr/bin/env bash
# Regenerate the Monte Carlo results by running each case's main1.py / main2.py.
#
#   scripts/run_all.sh                       # the ODEs (lorenz hopf pendulum), Parts I and II
#   scripts/run_all.sh --odes-burgers        # the ODEs and burgers
#   scripts/run_all.sh --all                 # every case, isothermal_flow included
#   scripts/run_all.sh --cases hopf lorenz   # only these cases
#   scripts/run_all.sh --parts 2             # only Part II (combine with any of the above)
#   scripts/run_all.sh --runs 25             # 25 Monte Carlo runs each (default: each config's, 100)
#   scripts/run_all.sh --jobs 8              # 8 runs in parallel in each case (default: each config's)
#
# Cases run one after the other; each Monte Carlo is parallel inside (cfg.n_jobs).
# A failing run is reported at the end and does not stop the others. Each run's
# output is also written to examples/<case>/results/main<part>.log.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ODES=(lorenz hopf pendulum)

cases=("${ODES[@]}")
parts=(1 2)
runs_args=()

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --odes)         cases=("${ODES[@]}"); shift ;;
        --odes-burgers) cases=("${ODES[@]}" burgers); shift ;;
        --all)          cases=("${ODES[@]}" burgers isothermal_flow); shift ;;
        --cases)        shift; cases=()
                        while [[ $# -gt 0 && "$1" != --* ]]; do cases+=("$1"); shift; done ;;
        --runs)         [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "--runs needs a number" >&2; exit 1; }
                        runs_args+=(--runs "$2"); shift 2 ;;
        --jobs)         [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "--jobs needs a number" >&2; exit 1; }
                        runs_args+=(--jobs "$2"); shift 2 ;;
        --parts)        shift; parts=()
                        while [[ $# -gt 0 && "$1" != --* ]]; do parts+=("$1"); shift; done ;;
        -h|--help)      usage ;;
        *)              echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

failed=()
for case in "${cases[@]}"; do
    for part in "${parts[@]}"; do
        dir="$ROOT/examples/$case"
        if [[ ! -f "$dir/main$part.py" ]]; then
            echo "!! $case: no main$part.py" >&2; failed+=("$case/main$part"); continue
        fi
        mkdir -p "$dir/results"
        echo "=== $case, Part $part ($(date +%H:%M:%S))"
        start=$SECONDS
        if (cd "$dir" && "$PYTHON" "main$part.py" ${runs_args[@]+"${runs_args[@]}"} 2>&1 | tee "results/main$part.log"); then
            echo "=== $case, Part $part done in $(( (SECONDS - start) / 60 )) min"
        else
            echo "!! $case, Part $part failed (see examples/$case/results/main$part.log)" >&2
            failed+=("$case/main$part")
        fi
    done
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "all done"
