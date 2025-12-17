##
## EPITECH PROJECT, 2025
## Delivery
## File description:
## Makefile
##

## ------------------------------------ ##
##              VARIABLES               ##

PYTHON              := python3.12
VENV_DIR            := .venv
VENV_PYTHON         := $(VENV_DIR)/bin/python
VENV_PIP            := $(VENV_DIR)/bin/pip

ANALYZER_EXEC       := my_torch_analyzer
GENERATOR_EXEC      := my_torch_generator
TEST_EXECUTABLE     := run_tests.py

SRCDIR              := src
TOOLSDIR            := tools
TESTDIR             := tests
GENERATOR_SRCDIR    := $(TOOLSDIR)/generator

MAIN_ANALYZER       := $(SRCDIR)/my_torch_analyzer.py
MAIN_GENERATOR      := $(GENERATOR_SRCDIR)/builder.py

REQUIREMENTS        := requirements.txt

RESET               := \033[0m
GREEN               := \033[32m
BLUE                := \033[34m
CYAN                := \033[36m
RED                 := \033[31m
YELLOW              := \033[33m

## ------------------------------------ ##
##                RULES                 ##

all: venv $(ANALYZER_EXEC) $(GENERATOR_EXEC)
	@echo -e "$(GREEN)[✔] Project setup successfully.$(RESET)"

# Create virtual environment
venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(REQUIREMENTS)
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo -e "$(CYAN)[➜] Creating virtual environment$(RESET)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo -e "$(CYAN)[➜] Installing dependencies$(RESET)"
	@$(VENV_PIP) install -q --upgrade pip
	@$(VENV_PIP) install -q -r $(REQUIREMENTS)
	@echo -e "$(GREEN)[✔] Virtual environment ready$(RESET)"

# Create analyzer executable wrapper
$(ANALYZER_EXEC): $(MAIN_ANALYZER) venv
	@echo -e "$(CYAN)[➜] Creating $(ANALYZER_EXEC)$(RESET)"
	@cp $(MAIN_ANALYZER) $(ANALYZER_EXEC)
	@chmod +x $(ANALYZER_EXEC)
	@echo -e "$(GREEN)[✔] Analyzer created: $(ANALYZER_EXEC)$(RESET)"

# Create generator executable
$(GENERATOR_EXEC): $(MAIN_GENERATOR) venv
	@echo -e "$(CYAN)[➜] Creating $(GENERATOR_EXEC)$(RESET)"
	@cp $(MAIN_GENERATOR) $(GENERATOR_EXEC)
	@chmod +x $(GENERATOR_EXEC)
	@echo -e "$(GREEN)[✔] Generator created: $(GENERATOR_EXEC)$(RESET)"

# Check Python syntax
check: venv
	@echo -e "$(CYAN)[➜] Checking Python syntax$(RESET)"
	@$(VENV_PYTHON) -m py_compile $(MAIN_ANALYZER) && echo -e "$(GREEN)[✔] Analyzer syntax OK$(RESET)" || echo -e "$(RED)[✘] Analyzer syntax error$(RESET)"
	@$(VENV_PYTHON) -m py_compile $(MAIN_GENERATOR) && echo -e "$(GREEN)[✔] Generator syntax OK$(RESET)" || echo -e "$(RED)[✘] Generator syntax error$(RESET)"
	@find $(SRCDIR) -name "*.py" -exec $(VENV_PYTHON) -m py_compile {} \; && echo -e "$(GREEN)[✔] All modules syntax OK$(RESET)"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.gcno" -delete
	@find . -type f -name "*.gcda" -delete
	@rm -f vgcore.*
	@echo -e "$(RED)[✘] Python cache files removed.$(RESET)"

fclean: clean
	@rm -rf $(VENV_DIR)
	@rm -f $(ANALYZER_EXEC) $(GENERATOR_EXEC)
	@echo -e "$(RED)[✘] Executables and virtual environment removed.$(RESET)"

re: fclean
	@$(MAKE) all --no-print-directory

## ------------------------------------ ##
##              UNIT TESTS              ##

tests_run: venv
	@echo -e "$(CYAN)[➜] Running unit tests$(RESET)"
	@if [ -f "$(TESTDIR)/$(TEST_EXECUTABLE)" ]; then \
		$(VENV_PYTHON) $(TESTDIR)/$(TEST_EXECUTABLE); \
	elif [ -d "$(TESTDIR)" ] && [ -n "$$(find $(TESTDIR) -name 'test_*.py')" ]; then \
		$(VENV_PYTHON) -m pytest $(TESTDIR) -v; \
	else \
		echo -e "$(YELLOW)[!] No tests found$(RESET)"; \
	fi

# Install development dependencies
dev: venv
	@echo -e "$(CYAN)[➜] Installing development dependencies$(RESET)"
	@$(VENV_PIP) install -q pytest pytest-cov pylint black flake8
	@echo -e "$(GREEN)[✔] Development dependencies installed$(RESET)"

# Run linter
lint: venv
	@echo -e "$(CYAN)[➜] Running linter$(RESET)"
	@$(VENV_PYTHON) -m flake8 $(SRCDIR) --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo -e "$(GREEN)[✔] Linting complete$(RESET)"

# Format code
format: venv
	@echo -e "$(CYAN)[➜] Formatting code$(RESET)"
	@$(VENV_PYTHON) -m black $(SRCDIR) $(TOOLSDIR) || echo -e "$(YELLOW)[!] black not installed, run 'make dev'$(RESET)"

.PHONY: all venv clean fclean re check tests_run dev lint format
