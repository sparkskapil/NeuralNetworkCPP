#pragma once
#include "NeuralNetwork.h"

namespace NeuralNetworks
{
	static class CNeuralNetworkFactory
	{
	public:
		static INeuralNetwork* GetNeuralNetwork(std::vector<int> const& NeuronCounts);
	};
}