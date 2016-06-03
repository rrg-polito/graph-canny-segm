all:
	g++ *.cpp `pkg-config --cflags opencv --libs opencv` -lm -std=c++11 -lboost_system -o main

