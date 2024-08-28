# Makefile for Streamlit Anaconda Project

# Variables
CONDA_ENV_NAME = streamlit_project
PYTHON_VERSION = 3.12

# Create Conda environment
create_env:
	conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION) -y

# Activate Conda environment
activate_env:
	conda activate $(CONDA_ENV_NAME)

# Install dependencies
install_deps:
	conda install -c conda-forge streamlit -y
	pip install -r requirements.txt

# Run Streamlit app
run:
	streamlit run app.py

# Clean up
clean:
	conda env remove -n $(CONDA_ENV_NAME)

# Full setup
setup: create_env activate_env install_deps

.PHONY: create_env activate_env install_deps run clean setup



