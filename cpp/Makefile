
INCLUDEFLAGS=-I../include -I../solvers/


all: example

example: example.cpp ../solvers/PSE/mobility.so  ../solvers/NBody/mobility.so 
	$(CXX) -O3 -std=c++17 $(DOUBLEPRECISION) $(INCLUDEFLAGS)  $^ -o $@ $(LAPACK_LIBS) 


clean:
	rm -f example
