##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Makefile
##

BIN = my_torch_analyzer

all:
	cp main.py $(BIN)
	chmod 755 $(BIN)

clean:
	rm -f $(BIN)

fclean: clean

re: all

test:
	PYTHONPATH=$(PWD) pytest -q