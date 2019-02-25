
//====================================================================

#include "Perceptron.hxx"
#include "Random.h"
#include <iostream>
#include <limits>

//====================================================================

using namespace NeuralNetworks;

//====================================================================

Perceptron::Perceptron(const int inputsCount)
{
	//Extra weight will be used as a bias
	for(int i=0; i<=inputsCount; i++)
	{
		weights_.push_back(Random::get());
	}

	//Set Default Learning Rate 0.01
	setLearningRate();
	//Set Default Tollerance 0.00001
	setTollerance();
}

//====================================================================

void Perceptron::setLearningRate(const double learningRate)
{
	learningRate_ = learningRate;
}

//====================================================================

void Perceptron::setTollerance(const Error tollerance)
{
	tollerance_ = tollerance;
}

//====================================================================

void Perceptron::fit(const TrainingSet& trainingData, const Targets& targets)
{
	Error totalError=std::numeric_limits<double>::max();
	trainingIterations_ = 0;
	while(totalError>tollerance_)
	{
		totalError = 0;
		auto target = targets.begin();
		for(auto& input:trainingData)
		{
			totalError+=std::abs(fit(input,*target));
			++target;
		}
		trainingIterations_++;
	}
}

//====================================================================


Predictions Perceptron::predict(const TestingSet& testingData)const
{
	Predictions predictions = std::make_shared<Outputs>();
	for(auto input:testingData)
	{
		predictions->push_back(predict(input));
	}
	return predictions;
}

//====================================================================

Error Perceptron::fit(const Inputs& inputs,const Target& target)
{
	Output actual = predict(inputs);
	Error error = target - actual;
	tuneWeights(error,inputs);
	return error;
}

//====================================================================

Output Perceptron::predict(const Inputs& inputs)const
{
	double weightedSum = computeWeightedSum(inputs);
	weightedSum += weights_.back();
	return activationFunction(weightedSum);
}

//====================================================================

Output Perceptron::activationFunction(const double weightedSum)const
{
	//Sign Function
	return weightedSum>=0 ? 1:0;
	//Fast Sigmoid
	//return weightedSum/(std::abs(weightedSum)+1);
}

//====================================================================

Output Perceptron::computeWeightedSum(const Inputs& inputs)const
{
	Output sum = 0;
	for(size_t i=0;i<inputs.size();i++)
	{
		sum += inputs[i]*weights_[i];
	}
	return sum;
}

//====================================================================

void Perceptron::tuneWeights(const Error& error, const Inputs& inputs)
{
	size_t i;
	for(i=0;i<inputs.size();i++)
	{
		weights_[i]+= error * inputs[i] * learningRate_;
	}
	weights_[i] += error * 1; //Last weight/Bias update;
}


//====================================================================
