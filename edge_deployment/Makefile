CXX = g++
CXXFLAGS = -shared -fPIC -O3 -Wall -std=c++17 -march=native -fopenmp

TARGET = libengine.so
SOURCES = engine.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
