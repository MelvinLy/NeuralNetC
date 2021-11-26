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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
