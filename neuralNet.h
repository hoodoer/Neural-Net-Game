//---------------------------------------------------------------------------
/*
  This simple 3 layer neural net class is based on a chapter in the book
  AI for Game Developers, by David Bourg and Glenn Seemann. Modifications 
  were made to this implementation to support my CIS Capstone course.
  It is based on supervised training, with a traditional feed-forward model
*/
//---------------------------------------------------------------------------

#include <iostream>
using namespace std;
#include <string>


#ifndef NEURALNET_H
#define NEURALNET_H

// This class implements the layers used in the neural network. 
// The parent-child relationship is such that the input layer
// is the parent to the hidden layer, and the hidden layer
// is the parent to the output layer. The input layer has
// no parent, and the output layer has no child. 
class NeuralNetworkLayer
{
 public:
  int		NumberOfNodes;
  int		NumberOfChildNodes;
  int		NumberOfParentNodes;
  double**	Weights;
  double**	WeightChanges;
  double*	NeuronValues;
  double*	DesiredValues;
  double*	Errors;
  double*	BiasWeights;
  double*	BiasValues;
  double	LearningRate;

  bool		LinearOutput;
  bool		UseMomentum;
  double	MomentumFactor;

  NeuralNetworkLayer*		ParentLayer;
  NeuralNetworkLayer*		ChildLayer;

  NeuralNetworkLayer();

  void	Initialize(int	NumNodes, NeuralNetworkLayer* parent, NeuralNetworkLayer* child);
  void	CleanUp(void);
  void	RandomizeWeights(void);
  void	CalculateErrors(void);
  void	AdjustWeights(void);	
  void	CalculateNeuronValues(void);
};




// Implements a 3-Layer neural network with one input layer, one hidden layer, and one output layer
class NeuralNetwork 
{
 public:
  NeuralNetworkLayer	InputLayer;
  NeuralNetworkLayer	HiddenLayer;
  NeuralNetworkLayer	OutputLayer;

  void	 Initialize(int nNodesInput, int nNodesHidden, int nNodesOutput);
  void	 CleanUp();
  void	 SetInput(int i, double value);
  double GetOutput(int i);
  void	 SetDesiredOutput(int i, double value);
  void	 FeedForward(void);
  void	 BackPropagate(void);
  int	 GetMaxOutputID(void);
  double CalculateError(void);
  void	 SetLearningRate(double rate);
  void	 SetLinearOutput(bool useLinear);
  void	 SetMomentum(bool useMomentum, double factor);
  void	 DumpData(string filename);
  void   ReadData(string filename);
};

#endif   // NEURALNET_H
