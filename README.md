# analysis_of_distributions

# Downloading virtualenv
python -m pip install virtualenv

# Creating a virtual environment
python -m virtualenv venv

# Activating a virtual environment
source venv/bin/activate

# Installing requirements
pip install -r requirements.txt

# Usage of the script
python main.py -h

# Plotting the results
python main.py -p

# Apple Silicon
Proceed to the next section if you use a Mac with Apple Silicon

# Downloading Python with Apple Silicon
Download Python3.11

# Downloading virtualenv
python3.11 -m pip install virtualenv

# Creating a virtual environment
python3.11 -m virtualenv venv

# Activating a virtual environment
source venv/bin/activate

# Downloading requirements, including Scipy, on Apple Silicon
python3-intel64 -m pip install -r requirements.txt

# Usage of the script
python3-intel64 main.py -h