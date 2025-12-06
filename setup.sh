#!/bin/bash

###################################
# BACKEND SETUP
###################################
cd "backend"
# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate
# Install dependencies
pip install --upgrade pip
pip install flask flask-cors torch torchvision pillow
# Run the backend server
python app.py &
BACKEND_PID=$!

###################################
# FRONTEND SETUP
###################################
cd ../frontend
# Install dependencies
npm install
# Run the frontend server
npm run start

# Cleanup: Kill backend server on script exit
kill $BACKEND_PID
