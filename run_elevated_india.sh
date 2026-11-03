#!/usr/bin/env bash
# Indian crowds shot from height. Sequential, one process, free tier 15 RPM.
# Safe to rerun: every stage caches per video/frame, so it resumes where it stopped.
#
# --drop-ground   keep metadata-elevated AND unclear; drop advertised walking
#                 tours/vlogs. Not --require-elevated: a station video that was
#                 "unclear" in its metadata was 10/10 elevated in reality.
# --min-persons 0 YOLO cannot see people from above (counted 2 in a frame holding
#                 ~200), so it records counts but must not gate.
# Gemini's per-frame vantage call is the real height filter, applied in Stage 6.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
LOG=notes/2026-10-19-elevated-india.log

run() {
  loc=$1; query=$2
  echo "=== $loc :: $query :: $(date '+%F %T') ===" | tee -a "$LOG"
  python -u stage1_search.py "$query" \
    --max-results 6 --drop-ground --min-persons 0 --rpm 15 \
    --full-pipeline --out "crowd_output_india/$loc" \
    --dataset dataset --location "$loc" >>"$LOG" 2>&1
  echo "--- $loc exit=$? at $(date '+%F %T') ---" | tee -a "$LOG"
}

run delhi-chandni-chowk          "Chandni Chowk Delhi aerial drone view crowd"
run mumbai-dadar-market          "Mumbai Dadar market crowd aerial drone view"
run varanasi-dashashwamedh-ghat  "Varanasi ghat crowd aerial drone view"
run prayagraj-kumbh-mela         "Kumbh Mela crowd aerial drone footage"
run mumbai-railway-station       "Mumbai railway station crowd from above overhead"
run kolkata-howrah-bridge        "Howrah Bridge Kolkata crowd aerial drone"
run jaipur-johari-bazaar         "Jaipur bazaar crowd aerial drone view"
run amritsar-golden-temple       "Golden Temple Amritsar crowd aerial drone view"

echo "=== ELEVATED BATCH FINISHED $(date '+%F %T') ===" | tee -a "$LOG"
