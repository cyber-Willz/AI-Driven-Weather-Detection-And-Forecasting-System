#!/bin/sh
set -e
cd "$(dirname "$0")/.."
cargo test --release
cargo run --release -- evaluate --out results/report.md
cargo run --release -- evaluate-eu --out results/report_eu.md
cargo run --release -- hurricane --out results/report_hurricane.md
