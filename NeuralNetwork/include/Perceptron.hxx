#pragma once
#include "Perceptron.h"

namespace NeuralNetworks
{
	class Perceptron: virtual public IPerceptron
	{
	public:
		Perceptron(const int);
		~Perceptron();
		void setLearningRate(const double = 0.01) override;
		void setTollerance(const Error tollerance = 0.0) override;

		void fit(const TrainingSet&, const Targets&) override;
		Predictions predict(const TestingSet&)const override;

	protected:
		Error fit(const Inputs&,const Target&) override;
		Output predict(const Inputs&)const override;

		Output activationFunction(const double weightedSum)const override;
		Output computeWeightedSum(const Inputs&)const override;
		void tuneWeights(const Error&, const Inputs&) override;

	private:
		Weights weights_;
		double learningRate_;
		Error tollerance_;
		unsigned trainingIterations_;
	};

	inline Perceptron::~Perceptron() = default;
}
