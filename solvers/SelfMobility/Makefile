PYTHON=python3
INCLUDE_FLAGS:= `$(PYTHON)-config --includes`   -I ../../include -I ../../third_party/pybind11/include $(LAPACK_INCLUDE)
CXX=g++
MODULENAME=SelfMobility
LIBRARY_NAME:=$(MODULENAME)$(shell $(PYTHON)-config --extension-suffix)

all: $(LIBRARY_NAME)

$(LIBRARY_NAME): python_wrapper.o 
	$(CXX) -g -O3 -w -shared   $< -o $@ $(LAPACK_LIBS)


python_wrapper.o: python_wrapper.cpp mobility.h
	$(CXX) -g -O3 -c -fPIC -std=c++14 $(DOUBLEPRECISION) $(INCLUDE_FLAGS) $<  -o $@ 


clean:
	rm -f $(LIBRARY_NAME) mobility.so python_wrapper.o
