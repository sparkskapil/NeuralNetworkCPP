#include "Matrix.h"
#include "Random.h"
#include <exception>
#include <fstream>
namespace NeuralNetworks
{
	///==================================================================================

	Matrix::Matrix(const unsigned rowCount,const unsigned colCount)
	{
		rows_ = rowCount;
		cols_ = colCount;
		mat_.resize(rowCount);
		for(auto& row:mat_)
		{
			row.resize(colCount);
			for(auto& cell : row)
			{
				cell = 0;
			}
		}
	}

	///==================================================================================

	Matrix::Matrix(std::vector<double> const& vector)
	{
		rows_ = 1;
		cols_ = vector.size();
		mat_.resize(rows_);
		for(unsigned i = 0;i<cols_;i++)
		{
			mat_[0].push_back( vector[i] );
		}
	}

	///==================================================================================

	Matrix::Matrix(Matrix const& matrix)
	{
		rows_ = matrix.rows_;
		cols_ = matrix.cols_;
		mat_.resize(rows_);

		for(unsigned i=0;i<rows_;i++)
		{
			mat_[i].resize(cols_);
			for(unsigned j=0;j<cols_;j++)
				mat_[i][j] = matrix.mat_[i][j];
		}
	}

	///==================================================================================

	Matrix Matrix::Transpose() const
	{
		Matrix transpose(this->cols_,this->rows_);
		for(unsigned i=0;i<rows_;i++)
			for(unsigned j=0;j<cols_;j++)
			{
				transpose.mat_[j][i] = mat_[i][j];
			}
		return transpose;
	}

	///==================================================================================

	Matrix Matrix::Add(Matrix const& matrix) const
	{
		if(this->rows_ != matrix.rows_ || this->cols_ != matrix.cols_ )
			throw std::runtime_error("Rows and Columns of the matrices are not equal");

		Matrix matSum(this->rows_,this->cols_);
		for(unsigned i=0;i<rows_;i++)
			for(unsigned j=0;j<cols_;j++)
			{
				matSum.mat_[i][j] = mat_[i][j] + matrix.mat_[i][j];
			}
		return matSum;
	}

	///==================================================================================

	Matrix Matrix::Multiply(Matrix const& matrix) const
	{
		if(this->cols_ != matrix.rows_)
			throw std::runtime_error("Rows and Columns of the matrices are not equal");

		Matrix matProduct(this->rows_,matrix.cols_);
		for(unsigned i=0;i<rows_;i++)
			for(unsigned j=0;j<cols_;j++)
			{
				for(unsigned k=0;k<matrix.cols_;k++)
					matProduct.mat_[i][k] += mat_[i][j] * matrix.mat_[j][k];
			}
		return matProduct;
	}

	///==================================================================================

	Matrix Matrix::Multiply(const double factor) const
	{
		Matrix matrix(rows_, cols_);
		for(unsigned i=0;i<rows_;i++)
			for(unsigned j=0;j<cols_;j++)
				matrix.mat_[i][j] = mat_[i][j]*factor;
		return matrix;
	}

	Matrix Matrix::Multiply(std::vector<double>&vector) const
	{
		if (vector.size() != rows_)
			throw "Incompatible Rows Size";
		Matrix result(rows_, cols_);
		for (int j = 0; j < cols_; j++)
			for (int i = 0; i < rows_; i++)
				result.mat_[i][j] = mat_[i][j] * vector[i];

		return result;
	}

	///==================================================================================

	void Matrix::Randomize()
	{
		for(unsigned i=0;i<rows_;i++)
			for(unsigned j=0;j<cols_;j++)
				mat_[i][j] = Random::get();
	}

	///==================================================================================

	Matrix Matrix::Identity(const unsigned size)
	{
		Matrix identity(size,size);
		for(unsigned i=0;i<size;i++)
			for(unsigned j=0;j<size;j++)
			{
				if(i==j)
					identity.mat_[i][j] = 1;
			}
		return identity;
	}

	///==================================================================================

	const unsigned Matrix::Rows() const
	{
		return rows_;
	}

	///==================================================================================

	const unsigned Matrix::Columns() const
	{
		return cols_;
	}

	///==================================================================================

	Matrix Matrix::operator+(Matrix const& matrix) const
	{
		return Add(matrix);
	}

	Matrix NeuralNetworks::Matrix::operator-(Matrix const & matrix) const
	{
		Matrix Negative = matrix * (-1);
		return Add(Negative);
	}

	///==================================================================================

	Matrix Matrix::operator*(Matrix const& matrix) const
	{
		return Multiply(matrix);
	}

	///==================================================================================

	Matrix Matrix::operator*(const double factor) const
	{
		return Multiply(factor);
	}

	Matrix Matrix::operator*(std::vector<double>&vector) const
	{
		return Multiply(vector);
	}

	///==================================================================================

	Matrix const& Matrix::operator=(Matrix const& matrix)
	{
		rows_ = matrix.rows_;
		cols_ = matrix.cols_;
		mat_.resize(rows_);

		for(unsigned i=0;i<rows_;i++)
		{
			mat_[i].resize(cols_);
			for(unsigned j=0;j<cols_;j++)
				mat_[i][j] = matrix.mat_[i][j];
		}
		return matrix;
	}

	///==================================================================================

	std::vector<double> const& Matrix::operator[](const unsigned row) const
	{
		return mat_[row];
	}

	///==================================================================================

	bool Matrix::operator ==(Matrix const& matrix)const
	{
		for(size_t i=0;i<rows_;i++)
			for(size_t j=0;j<cols_;j++)
				if(mat_[i][j] != matrix[i][j])
					return false;
		return true;
	}

	///==================================================================================

	bool Matrix::operator !=(Matrix const& matrix)const
	{
		return !this->operator==(matrix);
	}

	///==================================================================================

	Matrix Matrix::Map(std::function<double(double)> function) const
	{
		Matrix mat(rows_,cols_);
		for(size_t i=0;i<rows_;i++)
			for(size_t j=0;j<cols_;j++)
			{
				mat.mat_[i][j] = function(mat_[i][j]);
			}
		return mat;
	}

	///==================================================================================

	void Matrix::ToVector(std::vector<double>& vector ) const
	{
		int k=0;
		vector.resize(rows_*cols_);
		for(size_t i=0;i<rows_;i++)
		{
			for(size_t j=0;j<cols_;j++)
			{
				vector[k++] = mat_[i][j];
			}
		}
	}
	double Matrix::GetElementSum() const
	{
		double sum = 0;
		for (auto& row : mat_)
			for (auto cell : row)
				sum += cell;
		return sum;
	}
	double Matrix::GetAbsoluteSum() const
	{
		double sum = 0;
		for (auto& row : mat_)
			for (auto cell : row)
				sum += std::abs(cell);
		return sum;
	}

	double Matrix::GetAbsoluteMean() const
	{
		return GetAbsoluteSum()/(rows_*cols_);
	}

	void Matrix::SaveToFile(std::ofstream &out) const
	{
		out << rows_ <<","<< cols_;
		for (auto& row : mat_)
		{
			for (auto& cell : row)
				out << "," << cell;
		}
		out << '\n';
	}

	void Matrix::LoadFromFile(std::ifstream &) const
	{
	}

}



