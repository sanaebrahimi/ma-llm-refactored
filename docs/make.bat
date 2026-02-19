@ECHO OFF
set SOURCEDIR=.
set BUILDDIR=_build

if "%1"=="" goto help
sphinx-build -b %1 %SOURCEDIR% %BUILDDIR%/%1
exit /b 0

:help
echo.Please use `make.bat ^<target^>` where ^<target^> is one of
sphinx-build -M help %SOURCEDIR% %BUILDDIR%
