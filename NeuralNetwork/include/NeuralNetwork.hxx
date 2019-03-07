
#pragma once

#include "NeuralNetwork.h"
#include <map>

namespace NeuralNetworks
{
	class Matrix;
	class NeuralNetwork:public virtual INeuralNetwork
	{
	public:
		NeuralNetwork(std::vector<int> const& NeuronCounts);
		~NeuralNetwork();
		void FeedForward(Inputs const&) override;
		Error BackPropogate(Targets const&) override;
		
		void SetLearningRate(double = 0.1);
		void SetErrorThreshold(double = 0.01);
		void SetDynamicLearningRate(bool = false, double minLimit = 0.1, double maxLimit = 2);
	
		void fit(TrainingSet& trainingData,TargetLabels& labels);
		MultiPredictions predict(TestingSet& testingData);
		
		void PrintOutput()const;
		void Save(std::string const& fileName)const;
		void mutate(double mutationRate = 0.1);
		
	protected:
		double Activation(double)const override;
		double Derivative(double)const;
		void InitializeWeights() override;
		void UpdateLearningRate();
		Error calculateMeanError();

	private:
		Layers layers_;
		std::vector<Matrix *> weights_;
		std::map<int, double> indexVsErrorMap_;
		double learningRate_;
		double lrMinLimit_;
		double lrMaxLimit_;
		bool dynamicLREnabled_;

		size_t inputLayerIndex;
		size_t outputLayerIndex;

		Error meanError_;
		Error errorThreshold_;
		unsigned iterations_;
	};
}
