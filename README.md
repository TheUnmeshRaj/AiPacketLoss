# Video Call Quality Monitor with AI

This application monitors video call quality and uses AI to predict and improve call quality.

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

## Using the Application

1. **Start Monitoring**:
   - Select your network interface
   - Enter target host (or leave as "auto")
   - Set monitoring duration
   - Click "Start Monitoring"

2. **Train AI Model**:
   - After collecting some data, click "Train AI Model"
   - Wait for the training to complete

3. **View Analysis**:
   - Click "Show Analysis" to see:
     - Quality predictions
     - Feature importance
     - Improvement suggestions

4. **Download Data**:
   - Click "Download Data" to save monitoring results

## Features

- Real-time network monitoring
- AI-powered quality prediction
- Quality improvement suggestions
- Historical data analysis
- Data export functionality

## Requirements

- Windows/Linux/MacOS
- Python 3.8+
- Network interface with internet access 