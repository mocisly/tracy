TRACY_PUBLIC := ../../public
NVCC         := nvcc
CXX          := g++
CUPTI_INC    := /usr/local/cuda/include
CUPTI_LIB    := /usr/local/cuda/lib64

TRACY_SRCS   := $(TRACY_PUBLIC)/TracyClient.cpp
INCLUDES     := -I$(TRACY_PUBLIC) -I$(CUPTI_INC)
LIBS         := -L$(CUPTI_LIB) -lcuda -lcupti -lpthread -ldl

CXXFLAGS_REL   := -O2 -DTRACY_ENABLE
CXXFLAGS_DBG   := -g -O0 -DTRACY_ENABLE
NVCCFLAGS_REL  := -O2 -DTRACY_ENABLE
NVCCFLAGS_DBG  := -g -O0 -DTRACY_ENABLE

.PHONY: all debug clean

all: repro

debug: repro_debug

# Release build
repro: repro.cu tracy_client.o
	$(NVCC) $(NVCCFLAGS_REL) $(INCLUDES) -o $@ $< tracy_client.o $(LIBS)

tracy_client.o: $(TRACY_SRCS)
	$(CXX) $(CXXFLAGS_REL) $(INCLUDES) -c -o $@ $<

# Debug build (asserts enabled, no NDEBUG)
repro_debug: repro.cu tracy_client_debug.o
	$(NVCC) $(NVCCFLAGS_DBG) $(INCLUDES) -o $@ $< tracy_client_debug.o $(LIBS)

tracy_client_debug.o: $(TRACY_SRCS)
	$(CXX) $(CXXFLAGS_DBG) $(INCLUDES) -c -o $@ $<

clean:
	rm -f repro repro_debug tracy_client.o tracy_client_debug.o
