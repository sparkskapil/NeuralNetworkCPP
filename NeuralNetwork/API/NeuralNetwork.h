#pragma once
#include "NeuralNetsTypes.h"
namespace NeuralNetworks
{
	class INeuralNetwork
	{
	public:
		virtual ~INeuralNetwork() = 0;
		virtual void FeedForward(Inputs const&) = 0;
		virtual void BackPropogate(Targets const&) = 0;

	protected:
		virtual double Activation(double)const = 0;
		virtual void InitializeWeights() = 0;

	};
	inline INeuralNetwork::~INeuralNetwork() = default;
}
