CC = gcc
PY = python3
CFLAGS	= -O3 -g -Wall -Wno-unused-but-set-variable -Winline -std=c11 \
		  -Wextra -pedantic -Wmissing-prototypes -Wstrict-prototypes \
          -Wold-style-definition -Wno-unused-parameter
LIB = -lm


CSRCS = $(wildcard *.c)
COBJ = $(patsubst %.c,%.o,$(CSRCS))

all: kernel test_operators

test_operators: $(COBJ) $(CTESTS_OBJ)
	$(CC) $(CFLAGS) $(COBJ) -o $@ $(LIB)

kernel:
	$(PY) kernel.py

obj/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

test-nonuniform:
	./test_operators 4 11 \
	../../openfd/sbp/resources/bndopt/x42.txt \
 	../../openfd/sbp/resources/bndopt/xhat42.txt

test-uniform:
	./test_operators 4 11 \


clean: 
	rm -f *.o test_operators
