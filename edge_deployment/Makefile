# Makefile for Flash Engine (Edge Optimized)
CXX = g++
CXXFLAGS = -shared -fPIC -O3 -Wall -std=c++17 -fopenmp
LDFLAGS = -Wl,--no-as-needed -lgomp

# Detect Architecture
ARCH := $(shell uname -m)

ifeq ($(ARCH), x86_64)
    CXXFLAGS += -march=native
else ifneq (,$(filter $(ARCH), aarch64 arm64 armv7l))
    # ARM-specific optimizations (Pi 4 / Phone)
    CXXFLAGS += -march=armv8-a+simd -mfpu=neon-fp-armv8
endif

all: libengine.so

libengine.so: engine.cpp
	$(CXX) $(CXXFLAGS) engine.cpp -o libengine.so $(LDFLAGS)

clean:
	rm -f libengine.so
