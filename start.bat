@echo off
setlocal enabledelayedexpansion

echo.
echo ==========================================
echo       GhostMind -- Setup and Run
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ required.
    echo         Download from https://python.org
    pause
    exit /b 1
)

REM Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js 18+ required.
    echo         Download from https://nodejs.org
    pause
    exit /b 1
)

REM Setup .env
if not exist "backend\.env" (
    copy "backend\.env.example" "backend\.env" >nul
    echo.
    echo  [!] Created backend\.env
    echo  [!] Add your GEMINI_API_KEY to backend\.env
    echo  [!] Free key at: https://aistudio.google.com/app/apikey
    echo.
    notepad "backend\.env"
    echo  Press Enter once you have saved your API key...
    pause >nul
)

REM Backend setup
echo Setting up Python backend...
cd backend

if not exist "venv" (
    echo   Creating virtual environment...
    python -m venv venv
)

echo   Installing dependencies (first run may take a minute)...
call venv\Scripts\pip.exe install -q --upgrade pip
call venv\Scripts\pip.exe install -q -r requirements.txt

cd ..

REM Frontend setup
echo.
echo Setting up frontend...
cd frontend
call npm install --silent
cd ..

REM Launch backend in new window
echo.
echo Starting backend...
start "GhostMind - Backend" cmd /k "cd /d %CD%\backend && call venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait for backend
timeout /t 4 /nobreak >nul

REM Launch frontend in new window
echo Starting frontend...
start "GhostMind - Frontend" cmd /k "cd /d %CD%\frontend && npm run dev"

echo.
echo ==========================================
echo   GhostMind is running!
echo ==========================================
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo.
echo   Two windows have opened for backend and frontend.
echo   Close them to stop the servers.
echo.
pause
