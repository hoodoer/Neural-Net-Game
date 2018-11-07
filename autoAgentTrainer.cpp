/*******************************************************************
Course Project
Simple Neural Net application, separate net training program

Course: Artificial Intelligence
Professor: Dr. Alan Breitler

July 2010

Student:
Drew Kirkpatrick
dkirkpatrick2001@my.fit.edu

*Note, this program is not generally called on it's own,
I use it with a training script to handle the randomized training.
This script is called trainScript
*******************************************************************/



#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

#include <signal.h>
#include "neuralNet.h"
#include "math.h"
#include "timer.h"
#include "mathVector.h"




/**********************************************************************
             Global variables and declarations
*********************************************************************/
// This is the filename of the training 
// data set to use
string trainingDataSetFilename;

// This is the filename of the new neural net
// to create, or the existing neural net to
// modify
string brainFilename;


// Structure of neural net
#define INPUTNEURONS  4
#define OUTPUTNEURONS 1

// Number of neurons in
// the hidden layer is
// passed in as a commandline
// argument
int HIDDENNEURONS;


// The structure of the 
// neural network inputs
struct brainInputs
{
  float agentPosition;
  float boxColor;
  float boxAngle;
  float isThereABox;
};


// The output of the neural network
struct brainOutputs
{
  float movement;
};





void printUsageInfo()
{
  cout<<"Usage: "<<endl<<endl;
  cout<<"For training:"<<endl;
  cout<<"aiTrainer [trainingDataSetFilename] [numHiddenNodes] [brainFilename]"<<endl;
}






// Use the training data set to create or
// modify a neural network. 
void trainBrain()
{
  bool   existingBrain = false;
  int    i             = 0;
  double error         = 1;
  int    counter       = 0;
  int    lineCounter   = 0;

  // The neural network to use 
  // for training
  NeuralNetwork trainerBrain;

  ifstream trainingData(trainingDataSetFilename.c_str(), ios::in);
  ifstream testBrainFile;

  testBrainFile.open(brainFilename.c_str(), ios::in);
  testBrainFile.close();

  
  if(testBrainFile.fail())
    {
      cout<<"Starting a new neural net."<<endl;
      existingBrain = false;
    }
  else
    {
      cout<<"Modifying an existing neural net."<<endl;
      existingBrain = true;
    }

  if (!trainingData)
    {
      cout<<"Failed to open "<<trainingDataSetFilename<<endl;
      exit(1);
    }

  brainInputs  neuralInputData;
  brainOutputs neuralOutputData;

  if (existingBrain)
    {
      // Read in the existing neural net
      trainerBrain.ReadData(brainFilename);
    }
  else
    {
      // Initialize the new neural network
      trainerBrain.Initialize(INPUTNEURONS,
			      HIDDENNEURONS,
			      OUTPUTNEURONS);
    }

  trainerBrain.SetLearningRate(0.2);

  // Use momentum, can help sometimes avoid
  // local minima and maxima
  trainerBrain.SetMomentum(true, 0.9);

  while (!trainingData.eof())
    {
      // Read in inputs from training data
      trainingData>>neuralInputData.agentPosition;
      trainingData>>neuralInputData.boxColor;
      trainingData>>neuralInputData.boxAngle;
      trainingData>>neuralInputData.isThereABox;

      // Read in desired outputs from training data
      trainingData>>neuralOutputData.movement;

      while ((error > 0.05) && (counter < 50000))
	{
	  error = 0.0;
	  counter++;

	  // Set the neural network inputs to training data
	  trainerBrain.SetInput(0, neuralInputData.agentPosition); 
	  trainerBrain.SetInput(1, neuralInputData.boxColor); 
	  trainerBrain.SetInput(2, neuralInputData.boxAngle); 
	  trainerBrain.SetInput(3, neuralInputData.isThereABox); 

	  // Show the neural network the desired output
	  trainerBrain.SetDesiredOutput(0, neuralOutputData.movement);

	  // And now for the learning part
	  trainerBrain.FeedForward();
	  error += trainerBrain.CalculateError();
	  trainerBrain.BackPropagate();
	  lineCounter++;
	}
    }

  trainerBrain.DumpData(brainFilename);
  trainingData.close();
}





// The main function, for training 
// neural nets
int main(int argc, char** argv)
{
  cout<<"Starting neural network trainer."<<endl;

  if (argc < 4)
    {
      printUsageInfo();
      return 0;
    }

  trainingDataSetFilename = argv[1];
  HIDDENNEURONS           = atoi(argv[2]);
  brainFilename           = argv[3];
  
  cout<<endl;
  cout<<"Using dataset: "<<trainingDataSetFilename<<endl;
  cout<<"Building a network with "<<HIDDENNEURONS<<" hidden nodes."<<endl;
  cout<<"Saving the brain to file: "<<brainFilename<<endl<<endl;

  trainBrain();

  return 0;
}

