#pragma once
#include<vector>
#include<memory>
#include<functional>

namespace NeuralNetworks
{
	class Matrix final
	{
	public:
		Matrix(const unsigned rowCount, const unsigned colCount);
		Matrix(std::vector<double> const&);
		Matrix(Matrix const&);

		//ADDING MOVE SEMANTICS 
		Matrix(Matrix&&);	//Move Constructor
		Matrix& operator=(Matrix&&); // Move Assignment Operator

		Matrix Transpose() const;
		Matrix Add(Matrix const&) const;
		Matrix Multiply(Matrix const&) const;
		Matrix Multiply(const double) const;
		Matrix Multiply(std::vector<double>&)const;

		Matrix operator+(Matrix const&) const;
		Matrix operator-(Matrix const&) const;
		Matrix operator*(Matrix const&) const;
		Matrix operator*(const double) const;
		Matrix operator*(std::vector<double>&)const;
		Matrix const& operator=(Matrix const&);

		std::vector<double> const& operator[](unsigned)const;
		bool operator ==(Matrix const&)const;
		bool operator !=(Matrix const&)const;

		void Randomize();
		static Matrix Identity(const unsigned size);

		const unsigned Rows() const;
		const unsigned Columns() const;

		Matrix Map(std::function<double(double)> function) const;
		void ToVector(std::vector<double>&) const;

		double GetElementSum() const;
		double GetAbsoluteSum() const;
		double GetAbsoluteMean() const;

		void SaveToFile(std::ofstream&) const;
		void LoadFromFile(std::ifstream&);

		void mutate(double mutationRate);
	private:
		std::vector<std::vector<double>> mat_;
		unsigned rows_;
		unsigned cols_;
	};
}
