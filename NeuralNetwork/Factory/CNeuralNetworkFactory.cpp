#include "CNeuralNetworkFactory.h"
#include "NeuralNetwork.hxx"

using namespace NeuralNetworks;

INeuralNetwork* NeuralNetworks::CNeuralNetworkFactory::GetNeuralNetwork(std::vector<int> const& NeuronCounts)
{
	return new NeuralNetwork(NeuronCounts);
}
