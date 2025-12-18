##
## EPITECH PROJECT, 2025
## Delivery
## File description:
## Makefile
##

## ------------------------------------ ##
##              VARIABLES               ##

PYTHON              := python3

ANALYZER_EXEC       := my_torch_analyzer
GENERATOR_EXEC      := my_torch_generator
TEST_EXECUTABLE     := run_tests.py

SRCDIR              := src
TOOLSDIR            := tools
TESTDIR             := tests
GENERATOR_SRCDIR    := $(TOOLSDIR)/generator

MAIN_ANALYZER       := $(SRCDIR)/my_torch_analyzer.py
MAIN_GENERATOR      := $(GENERATOR_SRCDIR)/builder.py

RESET               := \033[0m
GREEN               := \033[32m
BLUE                := \033[34m
CYAN                := \033[36m
RED                 := \033[31m
YELLOW              := \033[33m

## ------------------------------------ ##
##                RULES                 ##

all: $(ANALYZER_EXEC) $(GENERATOR_EXEC)
	@echo -e "$(GREEN)[✔] Project setup successfully.$(RESET)"

# Create analyzer executable wrapper
$(ANALYZER_EXEC): $(MAIN_ANALYZER)
	@echo -e "$(CYAN)[➜] Creating $(ANALYZER_EXEC)$(RESET)"
	@cp $(MAIN_ANALYZER) $(ANALYZER_EXEC)
	@chmod +x $(ANALYZER_EXEC)
	@echo -e "$(GREEN)[✔] Analyzer created: $(ANALYZER_EXEC)$(RESET)"

# Create generator executable
$(GENERATOR_EXEC): $(MAIN_GENERATOR)
	@echo -e "$(CYAN)[➜] Creating $(GENERATOR_EXEC)$(RESET)"
	@cp $(MAIN_GENERATOR) $(GENERATOR_EXEC)
	@chmod +x $(GENERATOR_EXEC)
	@echo -e "$(GREEN)[✔] Generator created: $(GENERATOR_EXEC)$(RESET)"

# Check Python syntax
check:
	@echo -e "$(CYAN)[➜] Checking Python syntax$(RESET)"
	@$(PYTHON) -m py_compile $(MAIN_ANALYZER) && echo -e "$(GREEN)[✔] Analyzer syntax OK$(RESET)" || echo -e "$(RED)[✘] Analyzer syntax error$(RESET)"
	@$(PYTHON) -m py_compile $(MAIN_GENERATOR) && echo -e "$(GREEN)[✔] Generator syntax OK$(RESET)" || echo -e "$(RED)[✘] Generator syntax error$(RESET)"
	@find $(SRCDIR) -name "*.py" -exec $(PYTHON) -m py_compile {} \; && echo -e "$(GREEN)[✔] All modules syntax OK$(RESET)"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.gcno" -delete
	@find . -type f -name "*.gcda" -delete
	@rm -f vgcore.*
	@echo -e "$(RED)[✘] Python cache files removed.$(RESET)"

fclean: clean
	@rm -f $(ANALYZER_EXEC) $(GENERATOR_EXEC)
	@echo -e "$(RED)[✘] Executables removed.$(RESET)"

re: fclean
	@$(MAKE) all --no-print-directory

## ------------------------------------ ##
##              UNIT TESTS              ##

tests_run:
	@echo -e "$(CYAN)[➜] Running unit tests$(RESET)"
	@if [ -f "$(TESTDIR)/$(TEST_EXECUTABLE)" ]; then \
		$(PYTHON) $(TESTDIR)/$(TEST_EXECUTABLE); \
	elif [ -d "$(TESTDIR)" ] && [ -n "$$(find $(TESTDIR) -name 'test_*.py')" ]; then \
		$(PYTHON) -m pytest $(TESTDIR) -v; \
	else \
		echo -e "$(YELLOW)[!] No tests found$(RESET)"; \
	fi

# Run linter
lint:
	@echo -e "$(CYAN)[➜] Running linter$(RESET)"
	@$(PYTHON) -m flake8 $(SRCDIR) --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@echo -e "$(GREEN)[✔] Linting complete$(RESET)"

# Format code
format:
	@echo -e "$(CYAN)[➜] Formatting code$(RESET)"
	@$(PYTHON) -m black $(SRCDIR) $(TOOLSDIR) || echo -e "$(YELLOW)[!] black not installed$(RESET)"

.PHONY: all clean fclean re check tests_run lint format
