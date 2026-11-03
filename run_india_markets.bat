@echo off
REM Runs the crowd pipeline (stages 1..8) on Indian markets / bazaars.
REM 5 queries x 5 videos. Each query gets its own output folder so that
REM results.json / segments.csv / discovery.json are not overwritten.
REM Needs YOUTUBE_API_KEY and GEMINI_API_KEY in .env, and FFmpeg on PATH.
cd /d "%~dp0"

echo === Chandni Chowk Delhi market walking tour ===
python stage1_search.py "Chandni Chowk Delhi market walking tour" --max-results 5 --full-pipeline --out crowd_output_india\chandni_chowk

echo === Sarojini Nagar market Delhi crowd ===
python stage1_search.py "Sarojini Nagar market Delhi crowd" --max-results 5 --full-pipeline --out crowd_output_india\sarojini_nagar

echo === Crawford Market Mumbai walking tour ===
python stage1_search.py "Crawford Market Mumbai walking tour" --max-results 5 --full-pipeline --out crowd_output_india\crawford_market

echo === Commercial Street Bangalore shopping crowd ===
python stage1_search.py "Commercial Street Bangalore shopping crowd" --max-results 5 --full-pipeline --out crowd_output_india\commercial_street

echo === Laad Bazaar Hyderabad night market ===
python stage1_search.py "Laad Bazaar Hyderabad night market" --max-results 5 --full-pipeline --out crowd_output_india\laad_bazaar

echo Done. Results are in crowd_output_india\<query>\results.json and segments.csv
pause
