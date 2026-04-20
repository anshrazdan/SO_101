% Set Python environment to your project venv
pyenv("Version","C:\Users\anshr\Downloads\Robot\Robot_arm\SO_101\.venv\Scripts\python.exe")

% Verify Python path
pyrun("import sys; print(sys.executable)")

% Verify lerobot import
pyrun("import lerobot; print('lerobot ok from matlab')")

