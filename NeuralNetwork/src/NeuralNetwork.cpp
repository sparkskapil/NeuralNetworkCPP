#include "NeuralNetwork.hxx"
#include "Matrix.h"
#include "Random.h"
#include <iostream>
#include <fstream>
namespace NeuralNetworks
{
	using namespace std::placeholders;
	NeuralNetwork::NeuralNetwork(const std::vector<int>& NeuronCount)
	{
		layers_.resize(NeuronCount.size());
		size_t i = 0;
		for(auto count:NeuronCount)
		{
			if (i == layers_.size() - 1)
				layers_[i].resize(count);
			else
			{
				layers_[i].resize(count + 1);
				layers_[i][count] = 1;
			}
			i++;
		}
		InitializeWeights();

		inputLayerIndex = 0;
		outputLayerIndex = i-1;
		SetLearningRate();
		SetDynamicLearningRate();
		SetErrorThreshold();
	}

	NeuralNetwork::~NeuralNetwork()
	{
		for(auto &weight:weights_)
		{
			if(weight != nullptr)
			{
				delete weight;
				weight = nullptr;
			}
		}
	}

	void NeuralNetwork::InitializeWeights()
	{
		for(size_t i=0;i<layers_.size()-1;i++)
		{
			Matrix *weight = new Matrix(layers_[i+1].size(),layers_[i].size());
			weight->Randomize();
			weights_.push_back(weight);
		}
	}

	void NeuralNetwork::UpdateLearningRate()
	{
		if (dynamicLREnabled_ == false)return;
		double offset = meanError_ / (1 - errorThreshold_);
		learningRate_ = lrMinLimit_ + (lrMaxLimit_ - lrMinLimit_)*offset;
	}

	void NeuralNetwork::FeedForward(Inputs const& input)
	{
		//Update Input Layer
		Layer& inputLayer = layers_[inputLayerIndex];
		if(input.size()+1 != inputLayer.size())
			throw "Incorrect Input Size";
		for (size_t i = 0; i < input.size(); i++)
		{
			inputLayer[i] = input[i];
		}

		//FeedForward And Evaluate Hidden Nodes Values
		for(size_t i=0;i<layers_.size()-1;i++)
		{
			std::vector<double>& firstLayer = layers_[i];
			std::vector<double>& secondLayer = layers_[i+1];
			auto& weights = *weights_[i];
			Matrix layer1(firstLayer);
			Matrix layer2 = weights*layer1.Transpose();
			std::function<double(double)> mapper = std::bind(&NeuralNetwork::Activation,this,_1);
			layer2 = layer2.Map(mapper);
			layer2.ToVector(secondLayer);
		}
	}

	double NeuralNetwork::Activation(double x)const
	{
		return 1/(1+std::exp(-x));
	}

	double NeuralNetwork::Derivative(double activatedVal) const
	{
		return activatedVal * (activatedVal - 1);
	}

	void NeuralNetwork::BackPropogate(Targets const& targets)
	{
		Layer& output = layers_[outputLayerIndex];
		Matrix Target = targets;	
		Matrix Output = output;	
		Matrix currentError = (Target - Output).Transpose();
		meanError_ = currentError.GetAbsoluteMean();
		UpdateLearningRate();

		for (int i = outputLayerIndex-1; i >= 0; i--)
		{
			std::function<double(double)> mapper =
				std::bind(&NeuralNetwork::Derivative, this, _1);
			Matrix gradient = Output.Map(mapper);
			Matrix nextLayer = Matrix(layers_[i]);
			
			//Calculate Delta Weight
			std::vector<double> errors;
			currentError.ToVector(errors);
			Matrix temp = (gradient.Transpose() * errors)* learningRate_;
			Matrix deltaWeight = temp * nextLayer;
			
			currentError = weights_[i]->Transpose() * currentError;
			Output = Matrix(layers_[i]);

			//Update Weights
			*weights_[i] = *weights_[i] - deltaWeight;
		}
		//meanError_ /= iterations_;
	}
	void NeuralNetwork::SetLearningRate(double rate)
	{
		learningRate_ = rate;
	}
	void NeuralNetwork::SetErrorThreshold(double threshold)
	{
		errorThreshold_ = threshold;
	}
	void NeuralNetwork::SetDynamicLearningRate(bool status, double minLimit, double maxLimit)
	{
		lrMinLimit_ = minLimit;
		lrMaxLimit_ = maxLimit;
		dynamicLREnabled_ = status;
	}

	void NeuralNetwork::fit(TrainingSet & trainingData, TargetLabels& labels)
	{
		iterations_ = 1;
		do
		{
			int index = (int)(Random::get() * 10) % trainingData.size();
			Inputs &input = trainingData[index];
			Targets &target = labels[index];
			this->FeedForward(input);
			this->BackPropogate(target);
			++iterations_;
		} while (meanError_ > errorThreshold_);
	}
	MultiPredictions NeuralNetwork::predict(TestingSet & testingData)
	{
		MultiPredictions predictions = std::make_shared<TargetLabels>();
		for (auto& data : testingData)
		{
			this->FeedForward(data);			
			predictions->push_back(Targets(layers_[outputLayerIndex]));
		}
		return predictions;
	}
	
	void NeuralNetwork::PrintOutput() const
	{
		int i = 0;
		for (auto& neuron : layers_[0])
		{
			if(i<layers_[0].size()-1)
				std::cout << neuron << '\t';
		}
		std::cout << " ==> ";
		for (auto& layer : layers_[outputLayerIndex])
		{
			std::cout << layer << '\t';
		}
		std::cout << '\n';
	}
	void NeuralNetwork::Save(std::string const& fileName) const
	{
		std::ofstream writer(fileName, std::ios::app);
		for (auto weight : weights_)
		{
			weight->SaveToFile(writer);
		}
	}
}
