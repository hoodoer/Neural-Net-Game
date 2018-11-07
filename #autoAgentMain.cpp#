/*******************************************************************
Course Project
Simple Neural Net application

Course: Artificial Intelligence
Professor: Dr. Alan Breitler

July 2010

Student:
Drew Kirkpatrick
dkirkpatrick2001@my.fit.edu
*******************************************************************/


#include <GL/glut.h>
#include <GL/gl.h>
#include <iostream>
#include <iomanip>
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

// For converting radians to degrees
// My brain doesn't work in radians
#define RAD2DEG 57.29578


// Set to false to use a neural net
// to move the agent "sled" around.
// Set to true to use keyboard controls,
// which is simply for testing purposes
const bool manualControl = false;


enum BoxColor
  {
    BLUE,
    RED
  };

enum AgentMotion
  {
    LEFT,
    RIGHT,
    STOP
  };

enum BoxDirection
  {
    BOX_LEFT,
    BOX_RIGHT
  };



// Agent can only move left and right
// 100 starts the agent in the middle
// of the screen
float agentX = 100.0;

// How should the agent be moving?
// Only used in testing manualControl mode
AgentMotion agentMotion = STOP;


// Current position of the box
float boxX = 0.0;
float boxY = 0.0;


// Angle and direction to the box from the
// agent
float        boxAngle = 0.0;
BoxDirection boxDirection;


// Variable used to try to confuse
// the neural network with things
// it hasn't been trained with
// The windFactor allows you to
// blow the box around, so it
// doesn't fall straight down.
// You can set this with the keyboard
// buttons z, x, and c. 
// z is more left wind, x is stop all wind,
// and c is more right blowing wind. 
float windFactor       = 0.0;


// Variable used to try to confuse
// the neural network.
// The color shift factor controls
// how "perfect" the color of the box is. If the 
// shift factor is zero, the agent will
// see a perfect Red (1, 0, 0) rgb,
// and a perfect Blue (0, 0, 1) rgb.
// the neural net has been trained with
// perfect colors only. For the neural net, 
// -1.0 is red, and 1.0 is blue. By 
// increasing this shift factor, these
// numbers will converge. A mixture of red/blue
// of (1, 0, 1) rgb would be a colorShiftFactor of 1.0.
// A value of 0.5 will be more blue than red to the 
// neural net. It will be interesting to see
// what it does in this case. 
float colorShiftFactor = 0.0;


// Is there a box falling?
bool boxActive = false;



// Agent will try to avoid the red boxes, 
// but catch the blue ones
BoxColor boxColor;



// Timer used to smooth out animations
Timer animationTimer;



// Score variables
int numBluesCaught = 0;
int numBluesMissed = 0;
int numRedsHitBy   = 0;
int numRedsDodged  = 0;





// This file contains the 
// neural network values
// to be used. It is the "brain"
// so to speak
const string netFileName = "./brains/neuralNetwork.brain";


// And of course, we need a neural 
// network
NeuralNetwork boxAgent;










/**********************************************************************
             Miscellaneous support functions
*********************************************************************/

void printScore()
{
  cout<<endl<<endl;
  cout<<"Number of Blue boxes caught: " <<numBluesCaught <<endl;
  cout<<"Number of Blue boxes missed: " <<numBluesMissed <<endl;
  cout<<"Number of Red boxes hit by:  " <<numRedsHitBy   <<endl;
  cout<<"Number of Red boxes dodged:  " <<numRedsDodged  <<endl <<endl;
}













/**********************************************************************
             Calculation functions
*********************************************************************/


// This function simply scales the values of variables to 
// be used as input to the neural network
void setupNeuralNetInputs()
{
  // This function prepares the inputs for the neural net.
  // Inputs/outputs to/from the neural net range from -1.0 to 1.0

  static float codedAngle          = 0.0;
  static float codedAgentPositionX = 0.0;
  static float codedBoxColor       = 0.0;
  static float codedIsThereABox    = 0.0;

  // We should never see any angle 
  // greater than 92 degrees, so 
  // this is a safe divisor to make
  // sure we never see a scaled angle
  // greater than 1.0.
  codedAngle = boxAngle / 92.0;

  // The angle itself doesn't tell us
  // a direction to the box. If it's
  // to the left, make the angle negative
  if (boxDirection == BOX_LEFT)
    {
      codedAngle *= -1.0;
    }

  
  // agentPositionX represents the agents
  // position in the X axis, scaled to -1.0 to 1.0
  // -1 is all the way left, 0 is center, and 1.0 is
  // all the way to the right
  codedAgentPositionX = agentX;
  codedAgentPositionX -= 100.0;
  codedAgentPositionX /= 92.0;


  // This variable represents the 
  // color of the box. -1.0 is red,
  // and 1.0 is blue.
  // It will be interesting to see what
  // happens later when we change this 
  // value to give the neural net a 
  // "sorta" blue box. 
  if (boxColor == RED)
    {
      codedBoxColor = -1.0 + colorShiftFactor;
    }
  else
    {
      codedBoxColor = 1.0 - colorShiftFactor;;
    }

  // The agent needs a way to tell
  // if there is a box falling.
  // Set this variable to 1.0 if there
  // is a box falling, or -1.0 if there
  // isn't a box. 
  if (boxActive)
    {
      codedIsThereABox = 1.0;
    }
  else
    {
      codedIsThereABox = -1.0;
    }

  // If we're not in manual control mode,
  // push these inputs into the neural network
  if (!manualControl)
    {
      boxAgent.SetInput(0, codedAgentPositionX);
      boxAgent.SetInput(1, codedBoxColor);
      boxAgent.SetInput(2, codedAngle);
      boxAgent.SetInput(3, codedIsThereABox);
    }
}





// This simple function calculates
// the angle to the box from the agent
// using simple vector math. 
void calculateVectors()
{
  static CVec3 agent;
  static CVec3 box;
  
  static CVec3 upVector(0.0f, 1.0f, 0.0f);
  static CVec3 toBoxVector;

  agent.x = agentX;
  agent.y = 0.0;
  agent.z = 0.0;

  box.x = boxX;
  box.y = boxY;
  box.z = 0.0;

  // Create the vector from the agent
  // to the box
  toBoxVector = box - agent;

  // normalize the vectors
  toBoxVector.Normalize();
  upVector.Normalize();

  // Find the angle
  boxAngle = acosf(upVector.Dot(toBoxVector));

  // Convert it to degrees for easier reading
  // in debug statements
  boxAngle *= RAD2DEG;
  
  // Figure out if the box is to the left or right,
  // since the angle is unsigned on it's own, and
  // doesn't give an indication which direction 
  // the box is. 
  if (box.x <= agent.x)
    {
      boxDirection = BOX_LEFT;
    }
  else
    {
      boxDirection = BOX_RIGHT;
    }
}





// This function moves the box downwards,
// and checks for collision with the ground
// and the agent itself. The score is changed
// based on the collision detection.
// It also will apply the "wind" to the box
// to move it side to side. 
void animateBox()
{
  // The box moves downwards at a constant rate
  boxY -= 1.0;

  // The box will move side to side
  // depending on the "wind"
  boxX += windFactor;

  // Keep the box on the screen.
  if (boxX < 3)
    {
      boxX = 3;
    }

  if (boxX > 197)
    {
      boxX = 197;
    }

  // All of the collision detection code...
  // It's at the collision height of the agent platform...
  if (((boxY-3) <= 8) && (boxY > 0.0))
    {
      if ((fabs(boxX - agentX)) <= 11.0)
	{
	  switch (boxColor)
	    {
	    case BLUE:
	      cout<<"Agent caught the blue box!"<<endl;
	      numBluesCaught++;
	      break;

	    case RED:
	      cout<<"Agent got bombed!"<<endl;
	      numRedsHitBy++;
	      break;
	    }

	  boxActive = false;
	  printScore();
	}
    }

  // Ok, the agent wasn't underneath the falling box,
  // check for when it lands on the ground...
  if (boxY <= 0.0)
    {
      boxY = 0.0;
      switch (boxColor)
	{
	case BLUE:
	  cout<<"Agent missed a blue box!"<<endl;
	  numBluesMissed++;
	  break;
		  
	case RED:
	  cout<<"Agent dodged a red bomb!"<<endl;
	  numRedsDodged++;
	  break;
	}
	      
      boxActive = false;
      printScore();
    }
}



// Temporary function for
// controlling the "agent" with
// the keyboard, for program testing
// To use this, the boolean variable
// manualControl must be set to true,
// which will disable the neural network
// and only allow manual movement
void manualMoveAgent()
{
  switch(agentMotion)
    {
    case LEFT:
      agentX -= 1.0;

      if (agentX < 8.0)
	{
	  agentMotion = STOP;
	  agentX = 8.0;
	}
      break;

    case RIGHT:
      agentX += 1.0;

      if (agentX > 192.0)
	{
	  agentMotion = STOP;
	  agentX = 192.0;
	}
      break;

    default: 
      return;
    }
}



// This function takes the output
// from the neural network, and 
// translates it into movement
// of the agent graphic
void moveAgent(float movementValue)
{
  static float adjustedMovement;
  static const float movementFactor = 5.0;
  
  // Convert this output to a negative value
  // for left movement, and positive for
  // right movement. 0.5 should be sitting still
  adjustedMovement = movementValue - 0.5;


  // Make the movements a bit bigger
  agentX += adjustedMovement * movementFactor;


  // Make sure the agent stays on the screen
  if (agentX < 8.0)
    {
      agentX = 8.0;
    }

  if (agentX > 192.0)
    {
      agentX = 192.0;
    }
}



// This is where the magic happens. 
// Calculate the inputs to feed into
// the neural net, feed the network forward,
// and retrieve the output from the network.
void runNeuralNetwork()
{
  static float brainMovement; 

  // This will setup the 
  // variables to input into
  // the neural network
  setupNeuralNetInputs();
  
  // If we're not in manual
  // testing mode, run the neural net
  if (!manualControl)
    {
      boxAgent.FeedForward();
      brainMovement = boxAgent.GetOutput(0);
      moveAgent(brainMovement);
    }
}









/**********************************************************************
             OpenGL Graphics functions
*********************************************************************/


// Graphics and calculation stuff...
void graphicsInitialization()
{
  // Sets the display window color to grey
  glClearColor (0.5, 0.5, 0.5, 0.0);

  // Sets the projection parameters
  glMatrixMode (GL_PROJECTION);

  // Specifies the coordinate of the clipping planes
  gluOrtho2D (0.0, 200.0, 0.0, 150.0); 
}






// We drop these boxes above the agent, and see
// what it does.
void drawBox()
{
  // Set the color for the box. It will be either
  // blue or red, however both colors can have 
  // some of the other mixed in with it depending
  // on the colorShiftFactor, which can be set
  // by the user with the keyboard. 
  switch (boxColor)
    {
    case BLUE:
      glColor3f(colorShiftFactor, 0.0, 1.0);  
      break;

    case RED:
      glColor3f(1.0, 0.0, colorShiftFactor); 
      break;
    }


  // Filled box
  glRecti(boxX-3, boxY-3, boxX+3, boxY+3);
}


// This draws the simple rectangle that
// represents the agent. 
void drawAgent()
{
  // The "agent" box will be green
  glColor3f(0.0, 1.0, 0.0);

  // Filled box
  glRecti(agentX-8, 2, agentX+8, 8);
}



// Main display function called by GLUT. 
void displayFunc()
{
  // Clear the display window
  glClear (GL_COLOR_BUFFER_BIT); 

  // If there is a box falling,
  // draw it
  if (boxActive)
    {
      drawBox();
    }
  
  // Always draw the
  // agent "sled"
  drawAgent();

  glutPostRedisplay();
  glutSwapBuffers();
}










/**********************************************************************
             GLUT Input functions 
*********************************************************************/


// Right mouse button drops blue boxes, Left mouse button drops red ones
void mouseFunction(GLint button, GLint action, GLint xMouse, GLint yMouse)
{
  if (action == GLUT_DOWN)
    {
      if (!boxActive)
	{
	  switch (button)
	    {
	    case GLUT_LEFT_BUTTON:
	      boxColor = RED;
	      break;

	    case GLUT_RIGHT_BUTTON:
	      boxColor = BLUE;
	      break;

	    default:
	      return;
	    }

	  boxX = xMouse/2;
	  boxY = (fabs(yMouse - 300))/2;
	  boxActive = true;
	  animationTimer.reset();
	}
    }
}




// Simple keyboard callback. 
// ESC quits the program,
// a, s, and d control the color shifting,
// and z, x, and c control the wind
void keyboardFunction(unsigned char key, int x, int y)
{
  switch(key)
    {
    case 27: // Escape key
      cout<<"Exiting the program."<<endl;
      exit(0);
      break;

    case 97: // a key
      // shift color more central
      colorShiftFactor += 0.05;
      
      if (colorShiftFactor > 1.0)
	{
	  colorShiftFactor = 1.0;
	}

      cout<<"Color shift factor now: "<<colorShiftFactor<<endl;
      break;

    case 115: // s key
      // Reset colors to the extremes
      colorShiftFactor = 0.0;

      cout<<"Resetting color shift factor."<<endl;
      cout<<"Color shift factor now: "<<colorShiftFactor<<endl;
      break;

    case 100: // d key
      // shift color back to the extremes
      colorShiftFactor -= 0.05;
      
      if (colorShiftFactor < 0.0)
	{
	  colorShiftFactor = 0.0;
	}

      cout<<"Color shift factor now: "<<colorShiftFactor<<endl;
      
      break;

    case 122: // z key
      // shift "wind" to the left
      windFactor -= 0.1;

      if (windFactor < -1.5)
	{
	  windFactor = -1.5;
	}

      cout<<"Wind factor now: "<<windFactor<<endl;
      break;

    case 120: // x key
      // Recenter the "wind", so there is none
      windFactor = 0.0;

      cout<<"Resetting wind factor."<<endl;
      cout<<"Wind factor now: "<<windFactor<<endl;
      break;

    case 99: // c key
      // shift "wind" to the right
      windFactor += 0.1;

      if (windFactor > 1.5)
	{
	  windFactor = 1.5;
	}

      cout<<"Wind factor now: "<<windFactor<<endl;
      break;
    }
}





// This function is used when in manual mode
// so that the user can control the "agents"
// position manually for testing. It uses
// the arrow keys. Left moves the agent left,
// right to the right, and down will stop
// the agent. 
void arrowKeyFunction(int key, int x, int y)
{
  switch (key)
    {
    case GLUT_KEY_LEFT:
      agentMotion = LEFT;
      break;

    case GLUT_KEY_RIGHT:
      agentMotion = RIGHT;
      break;

    case GLUT_KEY_DOWN:
      agentMotion = STOP;
      break;
    }
}



      





/**********************************************************************
             Primary functions (Main and Idle)
*********************************************************************/

// The idle function is where all my calculations are called from,
// since GLUT controls the main loop itself. 
void idle()
{
  // Animate the whole deal at a constant
  // rate. Should keep the speeds 
  // consistent over a range of CPU
  // speeds
  if (animationTimer.total() >= 0.015)
    {
      animationTimer.reset();

      // In this mode, the keyboard
      // controls the position of the 
      // "agent", no neural net is 
      // used
      if (manualControl)
	{
	  manualMoveAgent();
	}
      
      // If there is a box,
      // we need to animate it,
      // and calculate the angle
      // to it to feed into the 
      // neural network
      if (boxActive)
	{
	  animateBox();
	  calculateVectors();
	}

      // And run the neural network
      // this won't do much if 
      // we're in manualControl,
      // but that's just a testing
      // mode used during development. 
      runNeuralNetwork();
    }
}




// The main function, sets up everything,
// and then starts the glut main loop.
// All my code is called from the idle
// function
int main(int argc, char** argv)
{
  cout<<"Starting the neural net simulator."<<endl;

  // If we're not running in manual
  // control mode, open up our saved
  // neural network that was created 
  // with the separate aiTrainer program
  if (!manualControl)
    {
      boxAgent.ReadData(netFileName);
    }

  // Move onto the GLUT intitialization stuff. 
  glutInit(&argc, argv);  
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowPosition(50, 100);
  glutInitWindowSize(400, 300);
  glutCreateWindow("Neural net box catcher/dodger");
  graphicsInitialization(); 


  // Register my GLUT callbacks for input,
  // display, and calculation
  glutMouseFunc(mouseFunction);
  glutSpecialFunc(arrowKeyFunction);
  glutKeyboardFunc(keyboardFunction);
  glutDisplayFunc(displayFunc); 
  glutIdleFunc(idle);

  // And.... go!
  glutMainLoop(); 

  return 0;
}

