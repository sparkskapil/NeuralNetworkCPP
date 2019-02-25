
#include "Random.h"
#include <cstdlib>
#include <ctime>
using namespace NeuralNetworks;

double Random::get()
{
	return random(-1,1);
}

double Random::random(int min, int max)
{
	static bool first = true;
	if(first)
	{
		srand(time(NULL));
		first = false;
	}

	double r = (double)rand() / (double)RAND_MAX;
	return min + r * (max - min);
}




