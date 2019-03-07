#include "Perceptron.hxx"
#include "Random.h"
#include <iostream>
#include <string>
#include <fstream>
#include "Matrix.h"
#include "NeuralNetwork.hxx"

using namespace NeuralNetworks;
///==================================================

///TODO:
///CREATE UTILITY TO LOAD AND SAVE DATASETS
///ADD SERIALIZATION FOR PERCEPTRON

///NOTE:
///ACTIVATION FUNCTION SHOULD HAVE THE TARGETED
///OUTPUT VALUES IN ITS RANGE

///==================================================

void TestingPerceptronGraphExample()
{
	Inputs inputs(2);
	TrainingSet trainingData(100);
	TestingSet testingData(2);
	Targets targets;
	Perceptron perceptron(inputs.size());

	for(size_t i=0;i<trainingData.size();i++)
	{
		inputs[0] = Random::get();
		inputs[1] = Random::get();
		double target = inputs[0]<inputs[1]? 1 : -1;
		trainingData[i].assign(inputs.begin(),inputs.end());
		targets.push_back(target);
	}

	inputs[0] = 0.1;
	inputs[1] = 0.2;
	testingData[0].assign(inputs.begin(),inputs.end());
	inputs[0] = 0.2;
	inputs[1] = 0.1;
	testingData[1].assign(inputs.begin(),inputs.end());

	perceptron.fit(trainingData,targets);
	Predictions predictions_training = perceptron.predict(trainingData);
	Predictions predictions_testing = perceptron.predict(testingData);

}

void TestingPerceptronANDExample()
{

	Inputs inputs(3);
	TrainingSet trainingData(6);
	TestingSet testingData(2);
	Targets targets;
	Perceptron perceptron(inputs.size());

	for(size_t i=0;i<8;i++)
	{
		inputs[0] = (double)((i >> 0) & 0x0001);
		inputs[1] = (double)((i >> 1) & 0x0001);
		inputs[2] = (double)((i >> 2) & 0x0001);

		double target = inputs[0] && inputs[1] && inputs[2];
		if(i>1)
		{
			trainingData[i-2].assign(inputs.begin(),inputs.end());
			targets.push_back(target);
			continue;
		}
		testingData[i].assign(inputs.begin(),inputs.end());
	}

	std::cout<<"Training Data\n";
	for(auto data:trainingData)
	{
		for(auto input:data)
		{
			std::cout<<input<<'\t';
		}
		std::cout<<'\n';
	}

	std::cout<<"Testing Data\n";
	for(auto data:testingData)
	{
		for(auto input:data)
		{
			std::cout<<input<<'\t';
		}
		std::cout<<'\n';
	}

	perceptron.fit(trainingData,targets);
	Predictions predictions_training = perceptron.predict(trainingData);
	Predictions predictions_testing = perceptron.predict(testingData);

	for(auto output: *predictions_training.get())
	{
		std::cout<<output<<'\n';
	}
	for(auto output: *predictions_testing.get())
	{
		std::cout<<output<<'\n';
	}
}

void PrintMatrix(Matrix& mat)
{
	std::cout<<'\n';
	for(size_t i = 0;i<mat.Rows();i++)
	{
		for(size_t j =0;j<mat.Columns();j++)
		{
			std::cout<< mat[i][j]<<' ';
		}
		std::cout<<'\n';
	}
}

void GetXORTrainingData(TrainingSet& trainingData, std::vector<Targets>& targetSet)
{
	Inputs inputs(2);
	Targets targets(1);
	inputs[0] = 0;
	inputs[1] = 0;
	targets[0] = 0;
	trainingData.push_back(Inputs(inputs));
	targetSet.push_back(targets);

	inputs[0] = 0;
	inputs[1] = 1;
	targets[0] = 1;
	trainingData.push_back(Inputs(inputs));
	targetSet.push_back(targets);

	inputs[0] = 1;
	inputs[1] = 0;
	targets[0] = 1;
	trainingData.push_back(Inputs(inputs));
	targetSet.push_back(targets);

	inputs[0] = 1;
	inputs[1] = 1;
	targets[0] = 0;
	trainingData.push_back(Inputs(inputs));
	targetSet.push_back(targets);

}

int NeuralNetsXOR()
{
	std::vector<int> neuronCounts = { 2,4,1 };
	NeuralNetwork nn(neuronCounts);
	Inputs inputs(2);
	Targets targets(1);
	char ch = ' ';

	TrainingSet trainingData;
	std::vector<Targets> targetSet;
	GetXORTrainingData(trainingData,targetSet);

	nn.SetLearningRate(0.2);
	nn.SetErrorThreshold(0.05);
	nn.fit(trainingData, targetSet);

	system("cls");
	inputs[0] = 0;
	inputs[1] = 0;
	nn.FeedForward(inputs);
	nn.PrintOutput();

	inputs[0] = 0;
	inputs[1] = 1;
	nn.FeedForward(inputs);
	nn.PrintOutput();

	inputs[0] = 1;
	inputs[1] = 0;
	nn.FeedForward(inputs);
	nn.PrintOutput();

	inputs[0] = 1;
	inputs[1] = 1;
	nn.FeedForward(inputs);
	nn.PrintOutput();
	std::cin.get();
	return 0;
}

using namespace std;
void CSVRowToVector(std::vector<double>&vector, string const &row)
{
	int i = 0;
	string value="";
	while (row[i] != '\0')
	{
		if (row[i] == ',')
		{
			vector.push_back(atof(value.c_str()));
			value = "";
		}
		else
			value += row[i];
		i++;
	}
}

void printVector(vector<double> const& vector,ofstream& writer)
{
	size_t size = vector.size();
	writer.write((char*)&size, sizeof(size_t));
	for (auto& val : vector) {
		auto data = val;
		writer << val << ",";
	}
	writer << '\n';
}
void printVector(vector<vector<double>>const& vector)
{
	ofstream writer("mnist_test_results.csv");
	for (auto& row : vector)
		printVector(row,writer);
	cout << '\n';
	writer.close();
}


int main()
{
	NeuralNetsXOR();
	return 0;
}