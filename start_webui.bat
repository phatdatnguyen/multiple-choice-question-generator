@echo off

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Create it first:
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate

if "%OPENAI_API_KEY%"=="" if not exist .env if not exist api_key.py (
    echo No API key found. Create a .env file containing:
    echo     OPENAI_API_KEY=sk-your-key-here
    pause
    exit /b 1
)

python webui.py
pause
