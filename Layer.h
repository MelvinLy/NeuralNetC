#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct Layer {
  double** weightMatrix;
  //Output size.
  int numRows;
  //Input size.
  int numCols;
  //Length of biases is the number of rows.
  double* biases;
} Layer;

double randFrom(double min, double max) 
{
		//Uncomment to use time as seed.
		//srand(time(NULL));
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

double sigmoidFunction(double x) {
  return (exp(x) / (exp(x) + 1));
}

double derivedSigmoidFunction(double x) {
  return (exp(x) / pow((exp(x) + 1), 2));
}

//Output array is the same size as input array.
//Free when not used.
double* applyNonLinearFunction(double* rawOutputVector, int len) {
  double* out = (double *) malloc(len * sizeof(double));
  for(int a = 0; a < len; a++) {
    out[a] = sigmoidFunction(rawOutputVector[a]);
  }
  return out;
}

double applyNonLinearFunctionS(double rawOutputValue) {
  return sigmoidFunction(rawOutputValue);
}

double applyDerivedNonLinearFunction(double rawOutputValue) {
  return derivedSigmoidFunction(rawOutputValue);
}

struct Layer createLayer(int inputSize, int outputSize) {
  Layer toReturn;
  toReturn.numRows = outputSize;
  toReturn.numCols = inputSize;
  toReturn.weightMatrix = (double**) malloc(outputSize * sizeof(double*));
  for(int a = 0; a < outputSize; a++) {
    toReturn.weightMatrix[a] = (double*) malloc(inputSize * sizeof(double));
  }
  toReturn.biases = (double*) malloc(outputSize * sizeof(double));
  for(int a = 0; a < outputSize; a++) {
    for(int b = 0; b < inputSize; b++) {
      toReturn.weightMatrix[a][b] = randFrom(-1, 1) * (double) sqrt(1.0 / inputSize);
    }
    toReturn.biases[a] = randFrom(-1, 1) * (double) sqrt(1.0 / inputSize);
  }
  return toReturn;
}

//Free when not used.
double* getRawOutput(Layer* layer, double* input) {
  double* out = (double*) malloc(layer->numRows * sizeof(double)); 
  for(int a = 0; a < layer->numRows; a++) {
    double* weightRow = layer->weightMatrix[a];
    double currentProduct = 0;
    for(int b = 0; b < layer->numCols; b++) {
      currentProduct = currentProduct + weightRow[b] * input[b];
    }
    out[a] = currentProduct + layer->biases[a];
  }
  return out;
}

double dCostByDRaw(double expectedValue, double rawValue) {
  return 2 * (applyNonLinearFunctionS(rawValue) - expectedValue) * applyDerivedNonLinearFunction(rawValue);
}
