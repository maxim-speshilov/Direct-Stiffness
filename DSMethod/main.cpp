#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

struct Element {
	std::pair <Eigen::SparseMatrix<float>, Eigen::VectorXf> DirectStiffnessMethod(float G, float tetta);
	Eigen::Matrix<float, 3, 6> B;
	int nodesIds[3];
};

std::vector <Element>       elements;
Eigen::VectorXf             nodesX;
Eigen::VectorXf             nodesY;
int                         nodesCount;


std::pair <Eigen::SparseMatrix<float>, Eigen::VectorXf> Element::DirectStiffnessMethod(float G, float tetta) {
	std::vector<Eigen::Triplet<float>> triplets;

	Eigen::Vector3f x, y;
	x << nodesX[nodesIds[0]], nodesX[nodesIds[1]], nodesX[nodesIds[2]];
	y << nodesY[nodesIds[0]], nodesY[nodesIds[1]], nodesY[nodesIds[2]];

	Eigen::Matrix3f C;
	C << Eigen::Vector3f(1.0f, 1.0f, 1.0f), x, y;

	float A = C.determinant() / 2.0;
	
	// B - gradient matrix
	
	Eigen::Matrix <float, 2, 3> B;

	B << y[1] - y[2], y[2] - y[0], y[0] - y[1],
		x[2] - x[1], x[0] - x[2], x[1] - x[0];

	B *= 1 / (2 *A);

	// stiffnes matrix of an element
		
	Eigen::Matrix<float, 3, 3> k = B.transpose() * B * A;

	for (int i = 0; i < 3; i++) 
		for (int j = 0; j < 3; j++) {
			Eigen::Triplet<float> triplet(nodesIds[i], nodesIds[j], k(i, j));
			triplets.push_back(triplet);
		}

	Eigen::SparseMatrix<float> K(nodesCount, nodesCount);
	K.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::VectorXf f(nodesCount);
	f.setZero();
	for (int i = 0; i < 3; i++)
		f(nodesIds[i], 0) = 2 * G * tetta * A / 3.0;

	
	std::pair <Eigen::SparseMatrix<float>, Eigen::VectorXf> result = std::make_pair(K, f);

	return result;
}

void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();

	if (rowToRemove < numRows)
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.bottomRows(numRows - rowToRemove);

	matrix.conservativeResize(numRows, numCols);
}

void calculate(){

	std::ifstream fin("input.txt");
	std::ifstream fin1("cond.txt");
	std::ofstream fout("output.txt");

	float G, tetta;

	fin >> G >> tetta;

	// getting nodes

	fin >> nodesCount;
	nodesX.resize(nodesCount);
	nodesY.resize(nodesCount);

	for (int i = 0; i < nodesCount; ++i)
		fin >> nodesX[i] >> nodesY[i];

	Eigen::VectorXf cond;
	int condCount;

	fin1 >> condCount;
	cond.resize(condCount);

	for (int i = 0; i < condCount; ++i)
		fin1 >> cond[i];


	// getting elements

	int elementCount;
	fin >> elementCount;

	for (int i = 0; i < elementCount; ++i) {
		Element element;
		fin >> element.nodesIds[0] >> element.nodesIds[1] >> element.nodesIds[2];
		elements.push_back(element);
	}

	Eigen::SparseMatrix<float> stiffnessMatrix;
	stiffnessMatrix.resize(nodesCount, nodesCount);
	stiffnessMatrix.setZero();
	Eigen::MatrixXf F(nodesCount,1);
	F.setZero();

	for (std::vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it){
		stiffnessMatrix += it->DirectStiffnessMethod(G, tetta).first;
		F += it->DirectStiffnessMethod(G, tetta).second;
	}

	fout << "F vector: \n" << F << std::endl;

	Eigen::MatrixXf m(stiffnessMatrix);

	for (int i = 0; i < condCount; i++){
		removeRow(m, cond[i]);
		removeRow(F, cond[i]);
	}

	Eigen::VectorXf phi(stiffnessMatrix.rows() - condCount);
	phi = m.colPivHouseholderQr().solve(F);

	fout << "Stiffness matrix: \n" << stiffnessMatrix << std::endl;
	fout << "Phi (solve): \n" << phi;

	fin.close();
	fin1.close();
	fout.close();
}

int main(){
	calculate();
	return 0;
}