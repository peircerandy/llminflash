# Makefile for Flash Engine (Edge Optimized)
CXX = g++
CXXFLAGS = -shared -fPIC -O3 -Wall -std=c++17 -fopenmp
LDFLAGS = -Wl,--no-as-needed -lgomp

# Detect Architecture
ARCH := $(shell uname -m)

ifeq ($(ARCH), x86_64)
    CXXFLAGS += -march=native
else ifneq (,$(filter $(ARCH), aarch64 arm64))
    # 64-bit ARM (Pi 4 / modern phones) - SIMD/Neon is mandatory
    CXXFLAGS += -march=armv8-a+simd
else ifeq ($(ARCH), armv7l)
    # 32-bit ARM (older Pi)
    CXXFLAGS += -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
endif

all: libengine.so

libengine.so: engine.cpp
	$(CXX) $(CXXFLAGS) engine.cpp -o libengine.so $(LDFLAGS)

clean:
	rm -f libengine.so
