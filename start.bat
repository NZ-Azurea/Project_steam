set "FOLDER=.venv"
rem
if not exist "%FOLDER%\" (
    echo Folder missing: "%FOLDER%"
    rem Example actions:
    uv venv .venv
    uv pip install -r requirements.txt
    echo Setup complete.
) else (
    echo Folder already exists: "%FOLDER%"
)
net start MongoDB
start "" cmd /k "cd /d "src" && uv run uvicorn API_DB:app --host 0.0.0.0 --port 27099 --reload"
start uv run streamlit run src/Application/app.py