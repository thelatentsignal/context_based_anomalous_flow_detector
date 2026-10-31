#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = context_based_anomalous_flow_detector
PYTHON_VERSION = 3.13.12
PYTHON_INTERPRETER = python3
#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: install
install: 
	@if command -v uv >/dev/null 2>&1; then \
		echo "Found uv. Syncing environment..."; \
		uv sync --extra dev; \
		echo "Registering Jupyter kernel..."; \
		.venv/bin/python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name "Python ($(PROJECT_NAME))"; \
	else \
		echo "uv not found. Falling back to standard pip with pinned versions..."; \
		python3 -m venv .venv; \
		.venv/bin/python -m pip install --upgrade pip; \
		.venv/bin/pip install -r requirements.txt; \
		.venv/bin/pip install -e ".[dev]"; \
     	.venv/bin/pip install ipykernel; \
		.venv/bin/python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name "Python ($(PROJECT_NAME))"; \
	fi


.PHONY: clean
clean:
	@echo "Cleaning up compiled Python files and build artifacts..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .venv
	@rm -rf build/ dist/ *.egg-info/ .eggs/
	@echo "Unregistering Jupyter kernel..."
	@if jupyter kernelspec list 2>/dev/null | grep -q "$(PROJECT_NAME)"; then \
		jupyter kernelspec remove -f "$(PROJECT_NAME)"; \
	else \
		echo "No Jupyter kernel named $(PROJECT_NAME) found"; \
	fi

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) context_based_anomalous_flow_detector/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
