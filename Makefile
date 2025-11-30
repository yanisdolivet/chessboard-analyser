##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Makefile
##

BIN = my_torch_generator

all:
	cp my_torch_generator.py $(BIN)
	chmod 755 $(BIN)

clean:
	rm -f $(BIN)

fclean: clean

re: all

test:
	PYTHONPATH=$(PWD) pytest -q