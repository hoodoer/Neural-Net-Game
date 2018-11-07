CC = g++



# Options passed to the compiler. -w suppresses warnings
OPTIONS = -O2


INCLUDES = \
	-I ./



# Various LIBS needed
LIBS = \
	-lglut                      \
	-lGLU                       \
	-lreadline


# Source code written for this project
SOURCES = \
	./autoAgentMain.cpp   \
	./neuralNet.cpp


# Used for building the training system
# for the neural network.
TRAINERSOURCES = \
	./autoAgentTrainer.cpp \
	./neuralNet.cpp



# The default, for building the simulation program
all:
	${CC} ${OPTIONS} ${INCLUDES} ${SOURCES} ${LIBS} -o autoAgent


# For building the training program
trainer:
	${CC} ${OPTIONS} ${INCLUDES} ${TRAINERSOURCES} ${LIBS} -o aiTrainer

