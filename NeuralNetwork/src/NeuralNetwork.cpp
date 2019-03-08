#include "NeuralNetwork.hxx"
#include "Matrix.h"
#include "Random.h"
#include <iostream>
#include <fstream>
#include <string>

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
			Matrix *weight = new Matrix((unsigned)layers_[i+1].size(),(unsigned)layers_[i].size());
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

	Error NeuralNetwork::calculateMeanError()
	{
		Error sum = 0;
		for (auto indexVsError : indexVsErrorMap_)
		{
			sum += indexVsError.second;
		}
		return sum/indexVsErrorMap_.size();
	}

	void NeuralNetwork::LoadLayers(std::ifstream &reader)
	{
		layers_.clear(); 
		while (reader.peek() != '\n')
		{
			int num;
			reader >> num;
			if (reader.peek() == ',')
				reader.ignore();
			Layer layer(num);
			layers_.push_back(layer);
		}
		reader.ignore();
	}

	void NeuralNetwork::LoadWeights(std::ifstream &reader)
	{
		for (auto& weight : weights_)
		{
			delete weight;
			weight = nullptr;
		}
		weights_.clear();

		std::string num = "";
		std::getline(reader,num);
		weights_.resize(std::stoi(num));
		size_t i = 0;
		while (i<weights_.size())
		{
			std::getline(reader, num,',');
			int rows = std::stoi(num);
			std::getline(reader, num,',');
			int cols = std::stoi(num);
			weights_[i] = new Matrix(rows, cols);
			weights_[i]->LoadFromFile(reader);
			i++;
		}
	}

	void NeuralNetwork::LoadInfo(std::ifstream &reader)
	{
		reader >> meanError_;
		reader >> errorThreshold_;
		reader >> iterations_;
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
			firstLayer[firstLayer.size() - 1] = 1;
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

	Error NeuralNetwork::BackPropogate(Targets const& targets)
	{
		Layer& output = layers_[outputLayerIndex];
		Matrix Target = targets;	
		Matrix Output = output;	
		Matrix currentError = (Target - Output).Transpose();
		double error = currentError.GetAbsoluteMean();
		UpdateLearningRate();

		for (int i = (int)outputLayerIndex-1; i >= 0; i--)
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
		return error;
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
			double error = this->BackPropogate(target);
			if (indexVsErrorMap_.find(index) == indexVsErrorMap_.end())
				indexVsErrorMap_.insert(std::make_pair(index, error));
			else
				indexVsErrorMap_[index] = error;
			meanError_ = calculateMeanError();
			++iterations_;
		} while (meanError_ > errorThreshold_ || indexVsErrorMap_.size() < trainingData.size());
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
		size_t i = 0;
		for (auto& neuron : layers_[0])
		{
			if(i<layers_[0].size()-1)
				std::cout << neuron << '\t';
			i++;
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

		//WRITE LAYER COUNTS IN FIRST ROW
		writer << "Layers Info";
		for (auto layer : layers_)
		{
			writer << ',' << layer.size();
		}
		writer << '\n';

		//WRITE WEIGHT MATRICES COUNT
		writer << "Weights Info," << weights_.size() <<'\n';

		//WRITE ALL WEIGHTS OF ALL MATRICES IN EACH ROW
		for (auto weight : weights_)
		{
			weight->SaveToFile(writer);
		}

		//WRITE TRAINING INFO
		writer << "Training Info," << meanError_ << ',' << errorThreshold_ << ',' << iterations_ << '\n';

		writer.close();
	}
	void NeuralNetwork::Load(std::string const & fileName)
	{
		std::ifstream reader(fileName);
		if (!reader.is_open())
			throw std::exception("File Not Opened!!!");
		std::string line;
		while (reader.good()) {
			std::getline(reader, line, ',');
			if (line == "Layers Info")
				LoadLayers(reader);
			else if (line == "Weights Info")
				LoadWeights(reader);
			else if (line == "Training Info")
				LoadInfo(reader);
		}
		reader.close();
	}
	void NeuralNetwork::mutate(double mutationRate)
	{
		for (auto weight : weights_)
		{
			weight->mutate(mutationRate);
		}
	}
}
