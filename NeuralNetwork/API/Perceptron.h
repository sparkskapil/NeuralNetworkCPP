#pragma once
#pragma warning(disable:4267)
#include "NeuralNetsTypes.h"

namespace NeuralNetworks
{
	class IPerceptron
	{
	public:
		virtual ~IPerceptron() = 0;
		virtual void setLearningRate(const double) = 0;
		virtual void setTollerance(const Error tollerance) = 0;
		virtual void fit(const TrainingSet&, const Targets&) = 0;
		virtual Predictions predict(const TestingSet&)const = 0;
	protected:

		virtual Error fit(const Inputs&, const Target&) = 0;
		virtual Output predict(const Inputs&)const = 0;
		virtual Output activationFunction(const double weightedSum)const = 0;
		virtual Output computeWeightedSum(const Inputs&)const = 0;
		virtual void tuneWeights(const Error&, const Inputs&) = 0;
	};
	inline IPerceptron::~IPerceptron() = default;
}
