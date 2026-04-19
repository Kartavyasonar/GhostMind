# GhostMind - PowerShell Startup Script
# Run with: .\start.ps1

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       GhostMind -- Setup and Run" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "OK  $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python 3.10+ is required." -ForegroundColor Red
    Write-Host "       Download from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node
try {
    $nodeVersion = node --version 2>&1
    Write-Host "OK  Node $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Node.js 18+ is required." -ForegroundColor Red
    Write-Host "       Download from https://nodejs.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Setup .env
if (-not (Test-Path "backend\.env")) {
    Copy-Item "backend\.env.example" "backend\.env"
    Write-Host ""
    Write-Host "  Created backend\.env" -ForegroundColor Yellow
    Write-Host "  Please open backend\.env and set your GEMINI_API_KEY" -ForegroundColor Yellow
    Write-Host "  Get a free key at: https://aistudio.google.com/app/apikey" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Press Enter once you have added your API key to backend\.env ..."
    Read-Host
}

# Backend venv
Write-Host ""
Write-Host "Setting up Python backend..." -ForegroundColor Cyan

Set-Location backend

if (-not (Test-Path "venv")) {
    Write-Host "  Creating virtual environment..."
    python -m venv venv
}

Write-Host "  Installing Python dependencies (this may take a minute on first run)..."
& "venv\Scripts\pip.exe" install -q --upgrade pip
& "venv\Scripts\pip.exe" install -q -r requirements.txt

Set-Location ..

# Frontend deps
Write-Host ""
Write-Host "Setting up frontend..." -ForegroundColor Cyan
Set-Location frontend
Write-Host "  Installing Node dependencies..."
npm install --silent
Set-Location ..

# Start backend in new window
Write-Host ""
Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; .\venv\Scripts\Activate.ps1; uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

# Brief wait for backend to boot
Write-Host "Waiting for backend to start..."
Start-Sleep -Seconds 4

# Start frontend in new window
Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  GhostMind is running!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend:   http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Two new PowerShell windows have opened for backend and frontend."
Write-Host "  Close them to stop the servers."
Write-Host ""
Read-Host "Press Enter to exit this window"
