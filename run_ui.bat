@echo off
title Legal AI Assistant - RAG v2 Web Server
cd /d "%~dp0"
echo ======================================================================
echo Starting Decepticons Legal AI Web Assistant (RAG v2)
echo ======================================================================
echo Launching server on http://localhost:8000 ...
start "" http://localhost:8000
.\legal-llm-venv\Scripts\python.exe ui\server.py 8000
pause
