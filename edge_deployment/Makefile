# Makefile for Flash Engine (Edge Optimized)
CXX = g++

# Detect Architecture
ARCH := $(shell uname -m)

# --- Memory-Aware Window Size Probing ---
# Probe System RAM (in MB) to calculate optimal Window Size (k)
# k=1 for <4GB (Phone), k=5 for 8GB (Pi/Laptop), k=10 for 16GB+
TOTAL_RAM := $(shell free -m | awk '/^Mem:/{print $$2}')
WINDOW_SIZE := 5
ifeq ($(shell expr $(TOTAL_RAM) \< 4000), 1)
    WINDOW_SIZE := 1
else ifeq ($(shell expr $(TOTAL_RAM) \> 12000), 1)
    WINDOW_SIZE := 10
endif

CXXFLAGS = -shared -fPIC -O3 -Wall -std=c++17 -fopenmp -DDEFAULT_WINDOW_SIZE=$(WINDOW_SIZE)
LDFLAGS = -Wl,--no-as-needed -lgomp

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
	@echo "--- Compilation Complete ---"
	@echo "Target Arch:  $(ARCH)"
	@echo "Detected RAM: $(TOTAL_RAM) MB"
	@echo "Window Size:  $(WINDOW_SIZE) tokens (Optimized for hardware)"

libengine.so: engine.cpp
	$(CXX) $(CXXFLAGS) engine.cpp -o libengine.so $(LDFLAGS)

clean:
	rm -f libengine.so
