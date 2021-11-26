#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "NeuralNetwork.h"

int main()
{
	NeuralNetwork nn = createNeuralNetwork(4);
	Layer l = createLayer(5, 5);
	addLayer(&nn, &l, 0);
	l = createLayer(5, 5);
	addLayer(&nn, &l, 1);
	l = createLayer(5, 5);
	addLayer(&nn, &l, 2);
	l = createLayer(5, 2);
	addLayer(&nn, &l, 3);

	double *inputs[] = {
	  (double[]) {0, 0, 1, 1, 1},
		(double[]) {1, 1, 1, 0, 0},
		(double[]) {1, 1, 1, 0, 0}
	};

	double *expectedOutputs[] = {
		(double[]) {0, 1},
		(double[]) {1, 0},
		(double[]) {1, 0},
	};

	fit(&nn, inputs, 3, expectedOutputs, 10000, 0.01);
	
	double* output = getOutputVector(&nn, inputs[2]);
	printArray(output, 2);
}