@echo off
setlocal EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem Run_DHRS7.bat -- opens the DHRS7 alignment window.
rem
rem Double-click this file. It looks for a Python that has the packages the
rem analysis needs, so it works without activating anything first. Plain
rem "python" is tried last, because on a machine with several Pythons that is
rem usually the system one, which does not have the packages.
rem
rem To force a particular interpreter, set PHYLO_PYTHON to its full path.
rem ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "PYEXE="

rem 1. explicit override
if defined PHYLO_PYTHON if exist "%PHYLO_PYTHON%" set "PYEXE=%PHYLO_PYTHON%"

rem 2. an environment that is already active
if not defined PYEXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PYEXE=%CONDA_PREFIX%\python.exe"

rem 3. an environment named phylo (or phylo_genes) under a usual conda root
if not defined PYEXE (
  for %%R in (
    "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
    "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3"
    "%ProgramData%\anaconda3" "%ProgramData%\miniconda3"
    "C:\Conda" "D:\Conda" "C:\ProgramData\Anaconda3"
  ) do (
    for %%E in (phylo phylo_genes) do (
      if not defined PYEXE if exist "%%~R\envs\%%E\python.exe" set "PYEXE=%%~R\envs\%%E\python.exe"
    )
  )
)

rem 4. whatever "python" happens to be
if not defined PYEXE set "PYEXE=python"

rem Use a local CA bundle if one is present (TLS-inspecting proxies).
if exist "%SCRIPT_DIR%ca_bundle.pem" (
    set "SSL_CERT_FILE=%SCRIPT_DIR%ca_bundle.pem"
    set "REQUESTS_CA_BUNDLE=%SCRIPT_DIR%ca_bundle.pem"
)

rem Confirm this interpreter can actually run the analysis before opening a window.
"%PYEXE%" -c "import Bio,pandas,numpy,matplotlib,requests" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Could not find a Python with the packages this analysis needs.
  echo.
  echo Tried: %PYEXE%
  echo.
  echo Create the environment once:
  echo     conda env create -f "%SCRIPT_DIR%phylo.yml"
  echo     conda activate phylo
  echo.
  echo Then run this file again, or set PHYLO_PYTHON to the full path of a
  echo Python that has biopython, pandas, numpy, matplotlib and requests.
  echo.
  pause
  exit /b 1
)

start "" "%PYEXE%" "%SCRIPT_DIR%MSA_GUI.py" %*
exit /b 0
