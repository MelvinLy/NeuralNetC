#include "Layer.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct NeuralNetwork {
  Layer* allLayers;
  int layerCount;
} NeuralNetwork;

struct NeuralNetwork createNeuralNetwork(int layerCount) {
  NeuralNetwork out;
  out.layerCount = layerCount;
  out.allLayers = (Layer*) malloc(layerCount * sizeof(Layer));
  return out;
}

void addLayer(NeuralNetwork* network, Layer* layer, int layerIndex) {
  network->allLayers[layerIndex] = *layer;
}

//Output size is the size of the output size of the final layer.
double* getOutputVector(NeuralNetwork* network, double* input) {
  double* out = input;
  for(int a = 0; a < network->layerCount; a++) {
    Layer currentLayer = network->allLayers[a];
    double* rawOutput = getRawOutput(&currentLayer, out);
    double* tmp = out;
    out = applyNonLinearFunction(rawOutput, currentLayer.numRows);
		if(tmp != input) {
    	free(tmp);
		}
    free(rawOutput);
  }
  return out;
}

double getCost(NeuralNetwork* network, double* predictedOutput, double* expectedOutput) {
  double out = 0;
  for(int a = 0; a < network->allLayers[network->layerCount - 1].numRows; a++) {
    out = out + pow(predictedOutput[a] - expectedOutput[a], 2);
  }
  return out;
}

void printArray(double* arr, int len) {
	for(int a = 0; a < len - 1; a++) {
		printf("%lf ", arr[a]);
	}
	printf("%lf\n", arr[len - 1]);
}

void fit(NeuralNetwork* network, double** inputs, int numInputs, double** expectedOutputs, int trainCycles, double learningRate) {
  for(int a = 0; a < trainCycles; a++) {
    //Free
    double*** adjustmentMatrices = (double***) malloc(network->layerCount * sizeof(double**));
    //Free
    double** biasAdjustmentMatrix = (double**) malloc(network->layerCount * sizeof(double*));
    for(int c = network->layerCount - 1; c >= 0; c--) {
      Layer currentLayer = network->allLayers[c];
      adjustmentMatrices[c] = (double**) malloc(currentLayer.numRows * sizeof(double*));
      for(int d = 0; d < currentLayer.numRows; d++) {
        adjustmentMatrices[c][d] = (double*) calloc(currentLayer.numCols, sizeof(double));
      }
      biasAdjustmentMatrix[c] = (double*) calloc(currentLayer.numRows, sizeof(double));
    }
    for(int b = 0; b < numInputs; b++) {
      double* currentInput = inputs[b];
      //Free
      double** allRawOutputs = (double**) malloc(network->layerCount * sizeof(double*));
      double* prev = currentInput;
      double** derivatives = (double**) malloc(network->layerCount * sizeof(double*));
      for(int c = 0; c < network->layerCount; c++) {
        Layer currentLayer = network->allLayers[c];
        //Free
        double* currentRawOutput = getRawOutput(&currentLayer, prev);
        allRawOutputs[c] = currentRawOutput;
        if(prev != currentInput) {
          free(prev);
        }
        //Free
        prev = applyNonLinearFunction(currentRawOutput, currentLayer.numRows);
      }
      double* predictedOutput = prev;
      derivatives[network->layerCount - 1] = (double*) calloc(network->allLayers[network->layerCount - 1].numRows, sizeof(double));
      Layer lastLayer = network->allLayers[network->layerCount - 1];
      for(int c = 0; c < network->allLayers[network->layerCount - 1].numRows; c++) {
        derivatives[network->layerCount - 1][c] = dCostByDRaw(expectedOutputs[b][c], allRawOutputs[network->layerCount - 1][c]);
      }
      free(prev);
      for(int c = network->layerCount - 1; c >= 0; c--) {
        Layer currentLayer = network->allLayers[c];
        double** currentWeightMatrix = currentLayer.weightMatrix;
        double** currentAdjustmentMatrix = adjustmentMatrices[c];
        double* currentDerivatives = derivatives[c];
        double* previousRawValues = NULL;
        Layer* previousLayer = NULL;
        if(c > 0) {
          previousRawValues = allRawOutputs[c - 1];
          previousLayer = &network->allLayers[c - 1];
        }
        else {
          previousRawValues = currentInput;  
        }
        double* currentBiasAdjustments = biasAdjustmentMatrix[c];
        for(int d = 0; d < currentLayer.numRows; d++) {
          double* currentRow = currentWeightMatrix[d];
          for(int e = 0; e < currentLayer.numCols; e++) {
            if(previousLayer != NULL) {
              currentAdjustmentMatrix[d][e] = currentAdjustmentMatrix[d][e] + currentDerivatives[d] * applyNonLinearFunctionS(previousRawValues[e]);
            }
            else {
              currentAdjustmentMatrix[d][e] = currentAdjustmentMatrix[d][e] + currentDerivatives[d] * previousRawValues[e];
            }
          }
          currentBiasAdjustments[d] = currentBiasAdjustments[d] + currentDerivatives[d];
        }
        if(previousLayer != NULL) {
          derivatives[c - 1] = (double*) calloc(currentLayer.numCols, sizeof(double));
          double* previousDerivatives = derivatives[c - 1];
          previousRawValues = allRawOutputs[c - 1];
          for(int d = 0; d < previousLayer->numRows; d++) {
            double previousDerivative = 0;
            for(int e = 0; e < currentLayer.numRows; e++) {
              int colVal = d;
              previousDerivative = previousDerivative + currentDerivatives[e] * currentWeightMatrix[e][colVal];
            }
            previousDerivatives[d] = previousDerivative * applyDerivedNonLinearFunction(previousRawValues[d]);
          }
        }
      }
      for(int z = 0; z < network->layerCount; z++) {
        free(derivatives[z]);
      }
      free(derivatives);
      for(int z = 0; z < network->layerCount; z++) {
        free(allRawOutputs[z]);
      }
      free(allRawOutputs);
    }
    for(int b = 0; b < network->layerCount; b++) {
      Layer currentLayer = network->allLayers[b];
      double** currentWeightMatrix = currentLayer.weightMatrix;
      double* currentBiases = currentLayer.biases;
      for(int c = 0; c < currentLayer.numRows; c++) {
        double* currentWeightRow = currentWeightMatrix[c];
        for(int d = 0; d < currentLayer.numCols; d++) {
          currentWeightRow[d] = currentWeightRow[d] - adjustmentMatrices[b][c][d] * learningRate / numInputs;
        }
        currentBiases[c] = currentBiases[c] - biasAdjustmentMatrix[b][c] * learningRate / numInputs;
      }
    }
    for(int z = 0; z < network->layerCount; z++) {
      for(int y = 0; y < network->allLayers[z].numRows; y++) {
        free(adjustmentMatrices[z][y]);
      }
      free(adjustmentMatrices[z]);
    }
    free(adjustmentMatrices);
    for(int z = 0; z < network->layerCount; z++) {
      free(biasAdjustmentMatrix[z]);
    }
    free(biasAdjustmentMatrix);
  }
}
