#!/usr/bin/env bash
# Install and verify the server's dependencies without creating application data.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo "This setup needs Linux x86_64. On Windows, run it inside Ubuntu on WSL2." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    case " $* " in
        *" --check "*) echo "Missing prerequisite: python3" >&2; exit 1 ;;
    esac
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Install Python 3.8+ first, then run this script again." >&2
        exit 1
    fi
    if [ "$(id -u)" = 0 ]; then
        apt-get update
        apt-get install -y python3
    else
        sudo apt-get update
        sudo apt-get install -y python3
    fi
fi

exec python3 "${PROJECT_DIR}/setup/server_dependencies.py" "$@"
