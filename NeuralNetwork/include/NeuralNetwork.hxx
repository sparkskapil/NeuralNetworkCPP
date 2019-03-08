
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
		
		void SetLearningRate(double = 0.1) override;
		void SetErrorThreshold(double = 0.01) override;
		void SetDynamicLearningRate(bool = false, double minLimit = 0.01, double maxLimit = 1.5)override;
	
		void fit(TrainingSet& trainingData,TargetLabels& labels) override;
		MultiPredictions predict(TestingSet& testingData) override;
		
		void PrintOutput()const;
		void Save(std::string const& fileName)const override;
		void Load(std::string const& fileName) override;
		void mutate(double mutationRate = 0.1) override;
		
	protected:

		double Activation(double)const;
		double Derivative(double)const;
		void InitializeWeights();
		void UpdateLearningRate();
		Error calculateMeanError();
		void LoadLayers(std::ifstream&);
		void LoadWeights(std::ifstream&);
		void LoadInfo(std::ifstream&);

	private:

		Layers layers_;
		std::vector<Matrix *> weights_;
		double learningRate_;

		size_t inputLayerIndex;
		size_t outputLayerIndex;

		double lrMinLimit_;
		double lrMaxLimit_;
		bool dynamicLREnabled_;

		Error meanError_;
		Error errorThreshold_;
		std::map<int, double> indexVsErrorMap_;
		unsigned iterations_;
	};
}
