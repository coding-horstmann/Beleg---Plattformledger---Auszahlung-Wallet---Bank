@echo off
cd /d "C:\Users\PC\Documents\Codex\2026-04-24\files-mentioned-by-the-user-daten"
"C:\Users\PC\Documents\Codex\2026-04-24\files-mentioned-by-the-user-daten\.venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false --server.fileWatcherType none
