#pragma once
namespace NeuralNetworks
{
	class Random
	{

	public:
		static double get();

	private:
		static double random(int min, int max);
	};
}
