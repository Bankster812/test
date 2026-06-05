@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   Keystone Property Partners -- One-Time Windows Setup
echo ============================================================
echo.

:: ── 1. Check for Python ──────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found on this machine.
    echo.
    echo Installing Python 3.12 via Windows Package Manager (winget)...
    echo This may take a minute. Please do not close this window.
    echo.
    winget install Python.Python.3.12 -e --silent --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Please install Python manually from:
        echo   https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    echo.
    echo Python installed successfully.
    echo Please CLOSE this window and run setup.bat again to continue.
    pause
    exit /b 0
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% found.
echo.

:: ── 2. Install optional Claude client ────────────────────────
echo Installing Claude client for AI-powered agent reasoning...
echo (This is optional -- the platform works without it using built-in logic)
echo.
python -m pip install anthropic --quiet --disable-pip-version-check
echo Done.
echo.

:: ── 3. Create Desktop shortcut ───────────────────────────────
echo Creating desktop shortcut...

set "SCRIPT=%~dp0launcher.pyw"
set "SHORTCUT=%USERPROFILE%\Desktop\Keystone Property Partners.lnk"

:: Find pythonw.exe (same directory as python.exe)
for /f "delims=" %%p in ('python -c "import sys,os; print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))"') do set PYTHONW=%%p

if not exist "!PYTHONW!" (
    :: Fallback: use python.exe if pythonw.exe not found
    for /f "delims=" %%p in ('python -c "import sys; print(sys.executable)"') do set PYTHONW=%%p
)

powershell -ExecutionPolicy Bypass -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; ^
     $s = $ws.CreateShortcut('!SHORTCUT!'); ^
     $s.TargetPath = '!PYTHONW!'; ^
     $s.Arguments = '\"!SCRIPT!\"'; ^
     $s.WorkingDirectory = '!CD!'; ^
     $s.IconLocation = 'shell32.dll,23'; ^
     $s.Description = 'Keystone Property Partners Dashboard'; ^
     $s.WindowStyle = 7; ^
     $s.Save()"

if errorlevel 1 (
    echo Warning: Could not create shortcut automatically.
    echo You can still launch the app by double-clicking launcher.pyw in this folder.
) else (
    echo Shortcut created on your Desktop.
)

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Double-click "Keystone Property Partners" on your Desktop
echo   to open the company dashboard.
echo.
echo   The dashboard will open automatically at:
echo   http://localhost:8200
echo ============================================================
echo.
pause
