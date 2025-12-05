##
## EPITECH PROJECT, 2025
## Delivery
## File description:
## Makefile
##

## ------------------------------------ ##
##              VARIABLES               ##

CC                  := g++
CFLAGS              := -std=c++17 -Wall -Wextra -pthread
INCLUDES            := -I./include -I./include/analyzer -I./include/my_torch
DFLAGS              := -g3
TFLAGS              := -lcriterion --coverage

ANALYZER_EXEC       := my_torch_analyzer
GENERATOR_EXEC      := my_torch_generator
TEST_EXECUTABLE     := unit_tests

OBJDIR              := obj
SRCDIR              := src
TOOLSDIR            := tools
TESTDIR             := tests
GENERATOR_SRCDIR    := $(TOOLSDIR)/generator

## Source files
ANALYZER_SRCDIR     := $(SRCDIR)/analyzer
MY_TORCH_SRCDIR     := $(SRCDIR)/my_torch

ANALYZER_SOURCES    := $(shell find $(ANALYZER_SRCDIR) -name '*.cpp')
MY_TORCH_SOURCES    := $(shell find $(MY_TORCH_SRCDIR) -name '*.cpp')
MAIN_SOURCE         := $(SRCDIR)/main.cpp
MAIN_GENERATOR		:= $(GENERATOR_SRCDIR)/builder.py

## Object files
ANALYZER_OBJECTS    := $(patsubst $(ANALYZER_SRCDIR)/%.cpp,\
						$(OBJDIR)/analyzer/%.o,$(ANALYZER_SOURCES))
MY_TORCH_OBJECTS    := $(patsubst $(MY_TORCH_SRCDIR)/%.cpp,\
						$(OBJDIR)/my_torch/%.o,$(MY_TORCH_SOURCES))
MAIN_OBJECT         := $(OBJDIR)/main.o


## Test files
TEST_SOURCES        := $(shell find $(TESTDIR) -name '*.cpp')
TEST_OBJECTS        := $(patsubst $(TESTDIR)/%.cpp,\
						$(OBJDIR)/tests/%.o,$(TEST_SOURCES))


RESET               := \033[0m
GREEN               := \033[32m
BLUE                := \033[34m
CYAN                := \033[36m
RED                 := \033[31m

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CFLAGS += $(DFLAGS)
endif

## ------------------------------------ ##
##                RULES                 ##

all: $(ANALYZER_EXEC)
	@echo "$(GREEN)[✔] Project compiled successfully.$(RESET)"

$(ANALYZER_EXEC): $(ANALYZER_OBJECTS) $(MY_TORCH_OBJECTS) $(MAIN_OBJECT)
	@mkdir -p $(@D)
	@echo "$(CYAN)[➜] Linking $(ANALYZER_EXEC)$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@
	cp $(MAIN_GENERATOR) $(GENERATOR_EXEC)
	chmod 755 $(GENERATOR_EXEC)
	@echo "$(GREEN)[✔] Analyzer compiled: $(ANALYZER_EXEC)$(RESET)"

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	@mkdir -p $(@D)
	@echo "$(BLUE)[~] Compiling: $<$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/analyzer/%.o: $(ANALYZER_SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "$(BLUE)[~] Compiling analyzer: $<$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/my_torch/%.o: $(MY_TORCH_SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "$(BLUE)[~] Compiling my_torch: $<$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/tests/%.o: $(TESTDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "$(BLUE)[~] Compiling test: $<$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)
	@rm -f $(TEST_EXECUTABLE)
	@rm -f *.gcno
	@rm -f *.gcda
	@rm -f vgcore.*
	@echo "$(RED)[✘] Objects and coverage files removed.$(RESET)"

fclean: clean
	@rm -f $(ANALYZER_EXEC) $(GENERATOR_EXEC) $(TEST_EXECUTABLE)
	@echo "$(RED)[✘] Executables removed.$(RESET)"

re: fclean
	@$(MAKE) all --no-print-directory

## ------------------------------------ ##
##              UNIT TESTS              ##

tests_run:
	@echo "$(CYAN)[➜] Compiling tests$(RESET)"
	@$(CC) $(CFLAGS) $(INCLUDES) $(TEST_SOURCES) $(ANALYZER_SOURCES) \
	$(MY_TORCH_SOURCES) $(TFLAGS) -o $(TEST_EXECUTABLE)
	@echo "$(GREEN)[✔] Unit tests executable created: $(TEST_EXECUTABLE)$(RESET)"
	@echo "$(CYAN)[➜] Running unit tests$(RESET)"
	@./$(TEST_EXECUTABLE)

coverage: tests_run
	@echo "$(CYAN)[➜] Generating code coverage report$(RESET)"
	@gcovr --exclude $(TESTDIR)

.PHONY: all clean fclean re tests_run coverage
