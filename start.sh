#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        GhostMind — Setup & Run           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3.10+ required. Install from https://python.org"
  exit 1
fi

# Check Node
if ! command -v node &>/dev/null; then
  echo "❌ Node.js 18+ required. Install from https://nodejs.org"
  exit 1
fi

# Backend setup
echo "📦 Setting up backend..."
cd backend

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  Created backend/.env — please add your GEMINI_API_KEY:"
  echo "   https://aistudio.google.com/app/apikey (free)"
  echo ""
  echo "   Edit backend/.env and set: GEMINI_API_KEY=your_key_here"
  echo ""
  read -p "Press Enter once you've set your API key..."
fi

if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "✅ Virtual environment created"
fi

source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Backend dependencies installed"

cd ..

# Frontend setup
echo ""
echo "📦 Setting up frontend..."
cd frontend
npm install --silent
echo "✅ Frontend dependencies installed"
cd ..

# Start both
echo ""
echo "🚀 Starting GhostMind..."
echo ""
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   API Docs: http://localhost:8000/docs"
echo ""

# Start backend in background
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ GhostMind is running!"
echo ""
echo "Press Ctrl+C to stop both servers."

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT

wait
