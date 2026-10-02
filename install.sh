#!/usr/bin/env bash
set -euo pipefail

./isaaclab.sh -p -m pip install -e source/tacex
./isaaclab.sh -p -m pip install -e source/tacex_assets
./isaaclab.sh -p -m pip install -e source/tacex_tasks
