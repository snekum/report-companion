#!/usr/bin/env bash
# Generates the synthetic population used by the pipeline.
# Pinned: Synthea v4.0.0, fixed seeds and a fixed reference date, so the
# same command reproduces the same records.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAVA="$ROOT/tools/jre/bin/java"
JAR="$ROOT/tools/synthea-v4.0.0.jar"
OUT="$ROOT/data/raw/synthea"
POP="${POP:-700}"
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$OUT"
"$JAVA" -Xmx4g -jar "$JAR" \
  -s 20261123 -cs 20261123 -r 20250601 -p "$POP" \
  --exporter.baseDirectory "$OUT" \
  --exporter.csv.export true \
  --exporter.fhir.export true \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false \
  --exporter.years_of_history 6 \
  --generate.only_alive_patients true \
  Massachusetts > "$OUT/synthea_run.log" 2>&1
tail -5 "$OUT/synthea_run.log"
