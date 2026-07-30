@echo off
if "%~1"=="-Z1" (
  tar -tf "%~2"
  exit /b %errorlevel%
)
if "%~1"=="-p" (
  tar -xOf "%~2" "%~3"
  exit /b %errorlevel%
)
echo Unsupported unzip arguments 1>&2
exit /b 2
