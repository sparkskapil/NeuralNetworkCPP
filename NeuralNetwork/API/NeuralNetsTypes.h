#pragma once
#include <vector>
#include <memory>


namespace NeuralNetworks
{
	typedef double Weight;
	typedef double Input;
	typedef double Output;
	typedef double Error;
	typedef double Target;

	typedef std::vector<double> Weights;
	typedef std::vector<double> Inputs;
	typedef std::vector<double> Outputs;
	typedef std::vector<double> Targets;

	typedef std::vector<Inputs> TrainingSet;
	typedef std::vector<Inputs> TestingSet;
	typedef std::vector<Targets> TargetLabels;

	typedef std::shared_ptr<Outputs> Predictions;
	typedef std::shared_ptr<TargetLabels> MultiPredictions;

	typedef double Neuron;
	typedef std::vector<Neuron> Layer;
	typedef std::vector<Layer> Layers;

}
