
#pragma once

#include "NeuralNetwork.h"

namespace NeuralNetworks
{
	class Matrix;
	class NeuralNetwork:public virtual INeuralNetwork
	{
	public:
		NeuralNetwork(std::vector<int> const& NeuronCounts);
		~NeuralNetwork();
		void FeedForward(Inputs const&) override;
		void BackPropogate(Targets const&) override;
		
		void SetLearningRate(double = 0.1);
		void SetErrorThreshold(double = 0.01);
		void SetDynamicLearningRate(bool = false, double minLimit = 0.1, double maxLimit = 2);
	
		void fit(TrainingSet& trainingData,TargetLabels& labels);
		MultiPredictions predict(TestingSet& testingData);
		
		void PrintOutput()const;
		void Save(std::string const& fileName)const;

	protected:
		double Activation(double)const override;
		double Derivative(double)const;
		void InitializeWeights() override;
		void UpdateLearningRate();

	private:
		Layers layers_;
		std::vector<Matrix *> weights_;
			
		double learningRate_;
		double lrMinLimit_;
		double lrMaxLimit_;
		double dynamicLREnabled_;

		size_t inputLayerIndex;
		size_t outputLayerIndex;

		Error meanError_;
		Error errorThreshold_;
		unsigned iterations_;
	};
}
