
#This Makefile probes the "solvers" folder and runs make inside each one
DIRECTORIES := $(sort $(dir $(wildcard solvers/*/)))

#export DOUBLEPRECISION=-DDOUBLE_PRECISION
export LAPACK_INCLUDE=
export LAPACK_LIBS=-llapacke -lcblas

all: $(DIRECTORIES) example_cpp

example_cpp:
	make -C cpp

clean: $(foreach dir,$(DIRECTORIES), clean_$(dir))

define makerule
.PHONY: $(1)
$(1):
	make -C $(1)

clean_$(1):
	make -C $(1) clean
endef

# Create a rule for all TEST_RUNNERS when "make test" is invoked...
$(foreach dir,$(DIRECTORIES),$(eval $(call makerule,$(dir))))

