
INCLUDEFLAGS=-I ../include
#This example must be compiled using the same precision as the particular library
#DOUBLEPRECISION=-DDOUBLE_PRECISION

all: example

example: example.cpp ../solvers/PSE/mobility.so  ../solvers/NBody/mobility.so 
	$(CXX) -O3 -std=c++14 $(DOUBLEPRECISION) $(INCLUDEFLAGS)  $^ -o $@


clean:
	rm -f example
