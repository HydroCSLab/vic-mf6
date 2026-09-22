#!/usr/bin/env bash

# expose one deterministic acceptance workflow from the installed model stack.

set -euo pipefail

install_dir=${VICMF6_INSTALL_DIR:-/opt/vicmf6}
coupler_source="$install_dir/src/vic-mf6"
example_source="$install_dir/examples/stehekin"

usage() {
    cat <<'EOF'
usage: container-entrypoint COMMAND [ARGUMENT]

commands:
  acceptance [OUTPUT_DIRECTORY]  run Stehekin and retain its products
  feedback-campaign OUTPUT_DIRECTORY
                                run the ten manuscript process experiments
  verification-evidence         check the compact H1--H8 evidence record
  versions                       print the installed component revisions
  shell                          open a diagnostic shell
EOF
}

write_environment() {
    local output_dir=$1
    {
        cat "$install_dir/share/component-revisions.txt"
        printf '\n'
        python --version
        mpirun --version
        mf6 -v
        python -m pip freeze
    } > "$output_dir/software-environment.txt"
}

collect_artifacts() {
    local example_dir=$1
    local output_dir=$2
    local status=$3
    local artifact

    if [ -d "$example_dir/run" ]; then
        cp -a "$example_dir/run/." "$output_dir/"
    fi
    for artifact in \
        exchange_table.csv \
        exchange_table.summary.json \
        exchange_table.summary.txt
    do
        if [ -f "$example_dir/$artifact" ]; then
            cp "$example_dir/$artifact" "$output_dir/$artifact"
        fi
    done

    if [ "$status" -eq 0 ]; then
        printf '%s\n' "PASS" > "$output_dir/acceptance-status.txt"
    else
        printf 'FAIL exit_code=%s\n' "$status" > "$output_dir/acceptance-status.txt"
    fi
    write_environment "$output_dir"
}

render_config() {
    local source_file=$1
    local output_file=$2

    python -E - "$source_file" "$output_file" "$install_dir" <<'PY'
import sys
from pathlib import Path

source_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
install_dir = sys.argv[3]
marker = "@INSTALL_DIR@"
template = source_file.read_text(encoding="utf-8")

if template.count(marker) != 3:
    raise SystemExit(f"expected three {marker} markers in {source_file}")

output_file.write_text(template.replace(marker, install_dir), encoding="utf-8")
PY
}

run_acceptance() {
    local output_dir=${1:-/results/stehekin}
    local work_dir
    local example_dir
    case "$output_dir" in
        /*) ;;
        *)
            printf '%s\n' "output directory must be absolute: $output_dir" >&2
            exit 2
            ;;
    esac

    mkdir -p "$output_dir"
    if [ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
        printf '%s\n' "output directory must be empty: $output_dir" >&2
        exit 2
    fi

    work_dir=$(mktemp -d /tmp/vic-mf6-acceptance.XXXXXX)
    cp -a "$coupler_source/." "$work_dir/"
    example_dir="$work_dir/examples/stehekin"
    cp -a "$example_source/." "$example_dir/"

    run_step() {
        local status

        if "$@"; then
            return 0
        else
            status=$?
            collect_artifacts "$example_dir" "$output_dir" "$status"
            rm -rf -- "$work_dir"
            printf '%s\n' "[FAIL] Stehekin acceptance; diagnostics: $output_dir" >&2
            exit "$status"
        fi
    }

    run_step render_config \
        "$example_source/config.yml" \
        "$example_dir/config.local.yml"
    run_step python -E "$example_dir/create_mf6.py"
    run_step "$work_dir/scripts/build_stehekin_exchange_table.sh" \
        "$example_dir/config.local.yml"
    run_step "$work_dir/vicmf6" inspect -c "$example_dir/config.local.yml"
    run_step mpirun --oversubscribe -np 2 \
        "$work_dir/vicmf6" run -c "$example_dir/config.local.yml"
    run_step python -E "$work_dir/scripts/check_stehekin_result.py" \
        "$example_dir/run/diagnostics"
    run_step "$work_dir/vicmf6" post all -c "$example_dir/config.local.yml"

    collect_artifacts "$example_dir" "$output_dir" 0
    rm -rf -- "$work_dir"
    printf '%s\n' "[OK] complete Stehekin acceptance workflow"
    printf '%s\n' "results: $output_dir"
}

command=${1:-acceptance}
case "$command" in
    acceptance)
        shift
        if [ "$#" -gt 1 ]; then
            usage >&2
            exit 2
        fi
        run_acceptance "${1:-/results/stehekin}"
        ;;
    feedback-campaign)
        shift
        if [ "$#" -ne 1 ]; then
            usage >&2
            exit 2
        fi
        case "$1" in
            /*) ;;
            *)
                printf '%s\n' "output directory must be absolute: $1" >&2
                exit 2
                ;;
        esac
        exec "$example_source/experiments/run-feedback-campaign.sh" "$1"
        ;;
    verification-evidence)
        if [ "$#" -ne 1 ]; then
            usage >&2
            exit 2
        fi
        exec python3 "$example_source/verification/verify_reference_results.py"
        ;;
    versions)
        if [ "$#" -ne 1 ]; then
            usage >&2
            exit 2
        fi
        cat "$install_dir/share/component-revisions.txt"
        ;;
    shell)
        if [ "$#" -ne 1 ]; then
            usage >&2
            exit 2
        fi
        exec /bin/bash
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
