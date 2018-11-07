#include "neuralNet.h"
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <fstream>

//---------------------------------------------------------------------------
/*
  This simple 3 layer neural net class is based on a chapter in the book
  AI for Game Developers, by David Bourg and Glenn Seemann. Modifications 
  were made to this implementation to support my CIS Capstone course.
  It is based on supervised training, with a traditional feed-forward model
*/
//---------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////////////////////////////////////
// NeuralNetworkLayer Class
/////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetworkLayer::NeuralNetworkLayer()
{
  ParentLayer    = NULL;
  ChildLayer     = NULL;
  LinearOutput   = false;
  UseMomentum    = false;
  MomentumFactor = 0.9;
}




// This function allocates the memory used by the neural network layer. 
void NeuralNetworkLayer::Initialize(int NumNodes, NeuralNetworkLayer* parent, NeuralNetworkLayer* child)
{
  int i, j;

  // Allocate memory
  NeuronValues  = (double*) malloc(sizeof(double) * NumberOfNodes);
  DesiredValues = (double*) malloc(sizeof(double) * NumberOfNodes);
  Errors = (double*) malloc(sizeof(double) * NumberOfNodes);

  if(parent != NULL)
    {		
      ParentLayer = parent;
    }

  if(child != NULL)
    {
      ChildLayer = child;

	
      Weights = (double**) malloc(sizeof(double*) * NumberOfNodes);
      WeightChanges = (double**) malloc(sizeof(double*) * NumberOfNodes);
      for(i = 0; i<NumberOfNodes; i++)
	{
	  Weights[i] = (double*) malloc(sizeof(double) * NumberOfChildNodes);
	  WeightChanges[i] = (double*) malloc(sizeof(double) * NumberOfChildNodes);
	}

      BiasValues = (double*) malloc(sizeof(double) * NumberOfChildNodes);
      BiasWeights = (double*) malloc(sizeof(double) * NumberOfChildNodes);
    } 
  else 
    {
      Weights     = NULL;
      BiasValues  = NULL;
      BiasWeights = NULL;
    }



  // Make sure everything contains zeros
  for(i=0; i<NumberOfNodes; i++)
    {
      NeuronValues[i] = 0;
      DesiredValues[i] = 0;
      Errors[i] = 0;
		
      if(ChildLayer != NULL)
	for(j=0; j<NumberOfChildNodes; j++)
	  {
	    Weights[i][j] = 0;
	    WeightChanges[i][j] = 0;
	  }
    }

  if(ChildLayer != NULL)
    for(j=0; j<NumberOfChildNodes; j++)
      {
	BiasValues[j] = -1;
	BiasWeights[j] = 0;
      }
}




// This function simply deallocates memory used
void NeuralNetworkLayer::CleanUp(void)
{
  int	i;

  free(NeuronValues);
  free(DesiredValues);
  free(Errors);
	
  if(Weights != NULL)
    {
      for(i = 0; i<NumberOfNodes; i++)
	{
	  free(Weights[i]);
	  free(WeightChanges[i]);
	}

      free(Weights);
      free(WeightChanges);
    }

  if(BiasValues != NULL) free(BiasValues);
  if(BiasWeights != NULL) free(BiasWeights);
}





// Called from initialize function for randomizing the weights
// of a new neural network. 
void NeuralNetworkLayer::RandomizeWeights(void)
{
  int	i,j;
  int	min = 0;
  int	max = 200;
  int	number;

  srand( (unsigned)time( NULL ) );

  for(i=0; i<NumberOfNodes; i++)
    {
      for(j=0; j<NumberOfChildNodes; j++)
	{	
	  number = (((abs(rand())%(max-min+1))+min));    
    
	  if(number>max)
	    number = max;

	  if(number<min)
	    number = min;		
			
	  Weights[i][j] = number / 100.0f - 1;
	}
    }
	
  for(j=0; j<NumberOfChildNodes; j++)
    {
      number = (((abs(rand())%(max-min+1))+min));    
    
      if(number>max)
	number = max;

      if(number<min)
	number = min;		
			
      BiasWeights[j] = number / 100.0f - 1;		
    }
}




// This function calculates the errors of specific neurons.
// It is called by the BackPropogate function for the overall
// neural network.
void NeuralNetworkLayer::CalculateErrors(void)
{
  int		i, j;
  double	sum;
	
  if(ChildLayer == NULL) // output layer
    {
      for(i=0; i<NumberOfNodes; i++)
	{
	  Errors[i] = (DesiredValues[i] - NeuronValues[i]) * NeuronValues[i] * (1.0f - NeuronValues[i]);
	}
    } 
  else if(ParentLayer == NULL) 
    { // input layer
      for(i=0; i<NumberOfNodes; i++)
	{
	  Errors[i] = 0.0f;
	}
    } 
  else 
    { // hidden layer
      for(i=0; i<NumberOfNodes; i++)
	{
	  sum = 0;
	  for(j=0; j<NumberOfChildNodes; j++)
	    {
	      sum += ChildLayer->Errors[j] * Weights[i][j];	
	    }
	  Errors[i] = sum * NeuronValues[i] * (1.0f - NeuronValues[i]);
	}
    }
}




// This function adjusts the weights of layers that have children,
// i.e. the input and hidden layers. Since the output layer doesn't
// have any connections to children, it doesn't apply here. In this 
// simple feed-forward style of neural network, each neuron in a layer
// connects to every neuron in it's child layer. 
void NeuralNetworkLayer::AdjustWeights(void)
{
  int		i, j;	
  double	dw;

  if(ChildLayer != NULL)
    {
      for(i=0; i<NumberOfNodes; i++)
	{
	  for(j=0; j<NumberOfChildNodes; j++)
	    {
	      dw = LearningRate * ChildLayer->Errors[j] * NeuronValues[i];
	      Weights[i][j] += dw + MomentumFactor * WeightChanges[i][j];			
	      WeightChanges[i][j] = dw;
	    }
	}

      for(j=0; j<NumberOfChildNodes; j++)
	{
	  BiasWeights[j] += LearningRate * ChildLayer->Errors[j] * BiasValues[j];
	}
    }
}



// This function is responsible for calculating the activation function
// of each neuron in the layer. 
// A logistic, or sigmoid activation function is used for all layers,
// except the output layer. The output layer will use a linear activation function
// if the boolean value LinearOutput is set to true. If that boolean is false,
// the output layer will also use the sigmoid activation function
void NeuralNetworkLayer::CalculateNeuronValues(void)
{
  int		i,j;
  double	x;
	
  if(ParentLayer != NULL)
    {
      for(j=0; j<NumberOfNodes; j++)
	{
	  x = 0;

	  for(i=0; i<NumberOfParentNodes; i++)
	    {
	      x += ParentLayer->NeuronValues[i] * ParentLayer->Weights[i][j];
	    }	
		
	  x += ParentLayer->BiasValues[j] * ParentLayer->BiasWeights[j];
			
	  if((ChildLayer == NULL) && LinearOutput)
	    {
	      // Linear activation function
	      NeuronValues[j] = x;
	    }
	  else
	    {
	      //  This is the logistic, or sigmoid activation function
	      NeuronValues[j] = 1.0f/(1+exp(-x));
	    }
	}
    }
}








/////////////////////////////////////////////////////////////////////////////////////////////////
// NeuralNetwork Class
/////////////////////////////////////////////////////////////////////////////////////////////////

// Called to initialize a new neural network. If you're starting with an existing
// neural net, you would call ReadData instead. The aiTrainer program would generally
// be calling this function to create and train new neural nets. Once a neural network
// has been created that you're happy with, you're end application would generally
// be calling ReadData instead, with the filename of the saved neural net. 
//
// It probably wouldn't be difficult to change this code to support multiple hidden
// layers, however most simple applications will not require more than one hidden layer.
// Indeed, many applications wouldn't even need a hidden layer, but this class is 
// explicitely a 3 layer network. Determining the correct topology in a multiple hidden
// layer network would be rather cumbersome trial and error effort. If such a 
// complicated neural network topology is needed, it may be more appropriate to 
// explore evolutionary methods of automatically evolving appropriate neural net
// topologies, using some implementation of the NEAT, rtNeat, or HyperNEAT algorithms 
void NeuralNetwork::Initialize(int nNodesInput, int nNodesHidden, int nNodesOutput)
{
  InputLayer.NumberOfNodes       = nNodesInput;
  InputLayer.NumberOfChildNodes  = nNodesHidden;
  InputLayer.NumberOfParentNodes = 0;	
  InputLayer.Initialize(nNodesInput, NULL, &HiddenLayer);
  InputLayer.RandomizeWeights();
	
  HiddenLayer.NumberOfNodes = nNodesHidden;
  HiddenLayer.NumberOfChildNodes = nNodesOutput;
  HiddenLayer.NumberOfParentNodes = nNodesInput;		
  HiddenLayer.Initialize(nNodesHidden, &InputLayer, &OutputLayer);
  HiddenLayer.RandomizeWeights();
	
  OutputLayer.NumberOfNodes       = nNodesOutput;
  OutputLayer.NumberOfChildNodes  = 0;
  OutputLayer.NumberOfParentNodes = nNodesHidden;		
  OutputLayer.Initialize(nNodesOutput, &HiddenLayer, NULL);	
}



// Cleans up all the layers of the neural network. 
void NeuralNetwork::CleanUp()
{
  InputLayer.CleanUp();
  HiddenLayer.CleanUp();
  OutputLayer.CleanUp();
}




// This function sets the input value for a specific
// input neuron. It is used both in training, and in
// the final application. 
void NeuralNetwork::SetInput(int i, double value)
{
  if((i>=0) && (i<InputLayer.NumberOfNodes))
    {
      InputLayer.NeuronValues[i] = value;
    }
}



// This function gets the output from a specific 
// output neuron. Called after the feedforward function
double NeuralNetwork::GetOutput(int i)
{
  if((i>=0) && (i<OutputLayer.NumberOfNodes))
    {
      return OutputLayer.NeuronValues[i];
    }

  return (double) INT_MAX; // to indicate an error
}



// This function is used during training to show the
// neural network the desired output. It uses this desired
// value, plus the value it actually calculated to determine
// an error rate
void NeuralNetwork::SetDesiredOutput(int i, double value)
{
  if((i>=0) && (i<OutputLayer.NumberOfNodes))
    {
      OutputLayer.DesiredValues[i] = value;
    }
}



// This function is called after setting the inputs to the
// input neurons. The feedforward function will 
// calculate all the values of each layer. After this
// function is done, the output neurons will contain 
// the output values. 
void NeuralNetwork::FeedForward(void)
{
  InputLayer.CalculateNeuronValues();
  HiddenLayer.CalculateNeuronValues();
  OutputLayer.CalculateNeuronValues();
}




// Used during training. The errors for the 
// output and hidden layers are calculated, 
// and then the weights are adjusted. 
void NeuralNetwork::BackPropagate(void)
{
  OutputLayer.CalculateErrors();
  HiddenLayer.CalculateErrors();

  HiddenLayer.AdjustWeights();
  InputLayer.AdjustWeights();
}



// This function is used to determine which
// output neuron has the highest output value.
// This would be useful in a neural net design 
// with a "winner-takes-all" approach, where
// you're looking for which output neuron 
// has the highest activation. 
int NeuralNetwork::GetMaxOutputID(void)
{
  int		i, id;
  double	maxval;

  maxval = OutputLayer.NeuronValues[0];
  id = 0;

  for(i=1; i<OutputLayer.NumberOfNodes; i++)
    {
      if(OutputLayer.NeuronValues[i] > maxval)
	{
	  maxval = OutputLayer.NeuronValues[i];
	  id = i;
	}
    }

  return id;
}



// This function is used during training to determine
// the error between the calculated output, and the
// desired output put forward by the training data. 
double NeuralNetwork::CalculateError(void)
{
  int		i;
  double	error = 0;

  for(i=0; i<OutputLayer.NumberOfNodes; i++)
    {
      error += pow(OutputLayer.NeuronValues[i] - OutputLayer.DesiredValues[i], 2);
    }

  error = error / OutputLayer.NumberOfNodes;

  return error;
}



// The learning rate for all layers are set to the
// same value with this function. 
void NeuralNetwork::SetLearningRate(double rate)
{
  InputLayer.LearningRate  = rate;
  HiddenLayer.LearningRate = rate;
  OutputLayer.LearningRate = rate;
} 



// Well, this silly function tries to set the value for
// all layers to use or don't use a linear activation 
// function, but only the output layer pays attention to it
// in this implementation. Something to fix going forward if 
// I want neural nets to be able to do something besides
// sigmoid activation functions for the input/hidden layer...
void NeuralNetwork::SetLinearOutput(bool useLinear)
{
  InputLayer.LinearOutput  = useLinear;
  HiddenLayer.LinearOutput = useLinear;
  OutputLayer.LinearOutput = useLinear;
}



// This function sets the useMomentum and momentum factor
// values for all layers in the network. Adding momentum can
// help alleviate the problem of hitting local minima/maxima. 
// The concepts is, with a little extra momentum to the weight
// adjustment, hoping that it can skip past local minima/maxima
void NeuralNetwork::SetMomentum(bool useMomentum, double factor)
{
  InputLayer.UseMomentum  = useMomentum;
  HiddenLayer.UseMomentum = useMomentum;
  OutputLayer.UseMomentum = useMomentum;

  InputLayer.MomentumFactor  = factor;
  HiddenLayer.MomentumFactor = factor;
  OutputLayer.MomentumFactor = factor;
}






// This dump data is not easily human read,
// however it's easy to parse by the software
// for reading in saved networks. 
void NeuralNetwork::DumpData(string filename)
{
  int i, j;
  ofstream brainFile(filename.c_str(), ios::out);

  brainFile<<InputLayer.NumberOfNodes<<endl;
  brainFile<<HiddenLayer.NumberOfNodes<<endl;
  brainFile<<OutputLayer.NumberOfNodes<<endl;

  // Added these to make sure you keep  fixed
  // point output. 9 decimal places should be 
  // more than enough. 
  brainFile.precision(9);
  brainFile.setf(ios::fixed);

  for(i=0; i<InputLayer.NumberOfNodes; i++)
    {
      brainFile<<InputLayer.NeuronValues[i]<<endl;
    }

  for(i=0; i<InputLayer.NumberOfNodes; i++)
    {
      for(j=0; j<InputLayer.NumberOfChildNodes; j++)
	{
	  brainFile<<i<<" "<<j<<" "<<InputLayer.Weights[i][j]<<endl;
	}
    }

  for(j=0; j<InputLayer.NumberOfChildNodes; j++)
    {
      brainFile<<j<<" "<<InputLayer.BiasWeights[j]<<endl;
    }

  for(i=0; i<HiddenLayer.NumberOfNodes; i++)
    {
      for(j=0; j<HiddenLayer.NumberOfChildNodes; j++)
	{
	  brainFile<<i<<" "<<j<<" "<<HiddenLayer.Weights[i][j]<<endl;
	}
    }

  for(j=0; j<HiddenLayer.NumberOfChildNodes; j++)
    {
      brainFile<<j<<" "<<HiddenLayer.BiasWeights[j]<<endl;
    }

  for(i=0; i<OutputLayer.NumberOfNodes; i++)
    {
      brainFile<<i<<" "<<OutputLayer.NeuronValues[i]<<endl;
    }

  brainFile.close();
}





// Call this with the name of a saved Neural
// net instead of calling initialize
void NeuralNetwork::ReadData(string filename)
{
  int i, j;
  int readI, readJ;

  ifstream brainFile(filename.c_str(), ios::in);

  brainFile>>InputLayer.NumberOfNodes;
  brainFile>>HiddenLayer.NumberOfNodes;
  brainFile>>OutputLayer.NumberOfNodes;

//   cout<<"Read in nodes: ("<<InputLayer.NumberOfNodes<<", "
//       <<HiddenLayer.NumberOfNodes<<", "
//       <<OutputLayer.NumberOfNodes<<")"<<endl;
  
  InputLayer.NumberOfChildNodes  = HiddenLayer.NumberOfNodes;
  InputLayer.NumberOfParentNodes = 0;
  InputLayer.Initialize(InputLayer.NumberOfNodes, NULL, &HiddenLayer);

  HiddenLayer.NumberOfParentNodes = InputLayer.NumberOfNodes;
  HiddenLayer.NumberOfChildNodes  = OutputLayer.NumberOfNodes;
  HiddenLayer.Initialize(HiddenLayer.NumberOfNodes, 
			 &InputLayer, &OutputLayer);

  OutputLayer.NumberOfParentNodes = HiddenLayer.NumberOfNodes;
  OutputLayer.NumberOfChildNodes  = 0;
  OutputLayer.Initialize(OutputLayer.NumberOfNodes, &HiddenLayer, NULL);
  


  for (i = 0; i < InputLayer.NumberOfNodes; i++)
    {
      brainFile>>InputLayer.NeuronValues[i];
    }
  
  for (i = 0; i < InputLayer.NumberOfNodes; i++)
    {
      for (j = 0; j < InputLayer.NumberOfChildNodes; j++)
	{
	  brainFile>>readI;
	  brainFile>>readJ;
	  if ((readI != i) || (readJ != j))
	    {
	      cout<<"Error, bad brainfile in readData 1!"<<endl;
	      brainFile.close();
	      exit(1);
	    }
	  brainFile>>InputLayer.Weights[i][j];
	}
    }

  for (i = 0; i < InputLayer.NumberOfChildNodes; i++)
    {
      brainFile>>readI;
      if (readI != i)
	{
	  cout<<"Error, bad brainfile in readData 2!"<<endl;
	  brainFile.close();
	  exit(1);
	}
      brainFile>>InputLayer.BiasWeights[i];
    }

  for (i = 0; i < HiddenLayer.NumberOfNodes; i++)
    {
      for (j = 0; j < HiddenLayer.NumberOfChildNodes; j++)
	{
	  brainFile>>readI;
	  brainFile>>readJ;
	  if ((readI != i) || (readJ != j))
	    {
	      cout<<"Error, bad brainfile in readData 3!"<<endl;
	      brainFile.close();
	      exit(1);
	    }
	  brainFile>>HiddenLayer.Weights[i][j];
	}
    }

  for (i = 0; i < HiddenLayer.NumberOfChildNodes; i++)
    {
      brainFile>>readI;
      if (readI != i)
	{
	  cout<<"Error, bad brainfile in readData 4!"<<endl;
	  cout<<"ReadI is: "<<readI<<" and i is: "<<i<<endl;
	  brainFile.close();
	  exit(1);
	}
      brainFile>>HiddenLayer.BiasWeights[i];
    }

  for (i = 0 ; i < OutputLayer.NumberOfNodes; i++)
    {
      brainFile>>readI;
      if (readI != i)
	{
	  cout<<"Error, bad brainfile in readData 5!"<<endl;
	  brainFile.close();
	  exit(1);
	}
      brainFile>>OutputLayer.NeuronValues[i];
    }

  brainFile.close();
}
