##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Makefile
##

CC	= g++

SRCDIR	=	src
TESTDIR	=	tests

BIN	=	my_torch_generator
MYTORCH	=	my_torch
TEST_EXECUTABLE     := unit_tests

SRCS	= 	src/main.cpp	\
	   		src/my_torch/Matrix.cpp	\

OBJS	=	$(SRCS:.cpp=.o)

TEST_SOURCES        := $(filter-out $(SRCDIR)/main.cpp, $(SRCS))
TEST_OBJECTS        := $(shell find $(TESTDIR) -name '*.cpp')
TEST_GENERATED_FILES := $(shell find . -name '*.gcno' -o -name '*.gcda')

CPPFLAGS	:= -I./include/my_torch -Wall -Wextra -Werror -std=c++17
DFLAGS	:= -g -fsanitize=address
TFLAGS	:= -lcriterion --coverage

all: $(MYTORCH)

$(MYTORCH): $(OBJS)
	@$(CC) $(CPPFLAGS) $^ -o $@
	cp ./tools/generator/builder.py $(BIN)
	chmod 755 $(BIN)

%/.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(BIN) $(MYTORCH)
	rm -f $(OBJS) $(TEST_EXECUTABLE)
	rm -f $(TEST_GENERATED_FILES)

fclean: clean

re: fclean
	@$(MAKE) all --no-print-directory

tests_run:
	@$(CC) $(CPPFLAGS) $(DFLAGS) $(TFLAGS) $(TEST_SOURCES) $(TEST_OBJECTS) -o $(TEST_EXECUTABLE)
	@./$(TEST_EXECUTABLE)

coverage:
	gcovr --exclude tests

test:
	PYTHONPATH=$(PWD) pytest -q

.PHONY: all clean fclean re tests_run test coverage