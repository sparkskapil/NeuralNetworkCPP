#pragma once
#include "NeuralNetsTypes.h"
namespace NeuralNetworks
{
	class INeuralNetwork
	{
	public:
		virtual ~INeuralNetwork() = 0;
		virtual void FeedForward(Inputs const&) = 0;
		virtual Error BackPropogate(Targets const&) = 0;
		virtual void SetLearningRate(double = 0.1) = 0;
		virtual void SetErrorThreshold(double = 0.01) = 0;
		virtual void SetDynamicLearningRate(bool = false, double minLimit = 0.01, double maxLimit=1.5)= 0;

		virtual void fit(TrainingSet& trainingData, TargetLabels& labels) = 0;
		virtual MultiPredictions predict(TestingSet& testingData) = 0;
		virtual void Save(std::string const& fileName)const = 0;
		virtual void Load(std::string const& fileName) = 0;
		virtual void mutate(double mutationRate = 0.1) = 0;

	};
	inline INeuralNetwork::~INeuralNetwork() = default;
}
