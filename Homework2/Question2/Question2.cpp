#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
using namespace std;

class DataPoint
{
private:
    double x1;
    double x2;
    double y;

public:
    DataPoint(double a, double b, double target)
    {
        x1 = a;
        x2 = b;
        y = target;
    }

    double getX1() const
    {
        return x1;
    }

    double getX2() const
    {
        return x2;
    }

    double getY() const
    {
        return y;
    }

    void print() const
    {
        cout << "x1: " << x1 
             << " | x2: " << x2
             << " | y: " << y << endl;
    }
};

class Dataset
{
private:
    vector<DataPoint> samples;

public:
    void addSample(const DataPoint& sample)
    {
        samples.push_back(sample);
    }

    int getSize() const
    {
        return static_cast<int>(samples.size());
    }

    const DataPoint& getSample(int index) const
    {
        return samples.at(index);
    }

    void printSamples(int count) const
    {
        int limit = count;

        if (limit > getSize())
        {
            limit = getSize();
        }

        for (int i = 0; i < limit; i++)
        {
            cout << "Sample " << i + 1 << ": ";
            samples[i].print();
        }
    }

    void saveCSV(const string& filename) const
    {
        ofstream file(filename);

        if (!file.is_open())
        {
            cerr << "Error opening " << filename << endl;
            return;
        }

        file << "x1,x2,y\n";
        file << setprecision(17);

        for (int i = 0; i < getSize(); i++)
        {
            const DataPoint& sample = samples[i];

            file << sample.getX1() << ","
                 << sample.getX2() << ","
                 << sample.getY() << "\n";
        }

        file.close();

        cout << "Saved: " << filename << endl;
    }
};

class GaussianMixtureGenerator
{
private:
    mt19937 rng;
    uniform_real_distribution<double> uniformDist;
    normal_distribution<double> normalDist;

public:
    GaussianMixtureGenerator(unsigned int seed = 42)
        : rng(seed), uniformDist(0.0, 1.0), normalDist(0.0, 1.0)
    {    
    }

    DataPoint generateSample()
    {
        // Select Gaussian component
        double u = uniformDist(rng);

        double z1 = normalDist(rng);
        double z2 = normalDist(rng);
        double z3 = normalDist(rng);

        double x1;
        double x2;
        double y;

        if (u < 0.3)
        {
            // Component 1
            // Mean: (-10, 0, 10)
            x1 = -10.0 + z1;
            x2 = z2;
            y = 10.0 - 3.0 * z1 + sqrt(6.0) * z3;
        }
        else if (u < 0.7)
        {
            // Component 2
            // Mean: (0, 0, 0)
            x1 = sqrt(8.0) * z1;
            x2 = sqrt(8.0) * z2;
            y = sqrt(0.5) * z3;
        }
        else
        {
            // Component 3
            // Mean: (10, 0, -10)
            x1 = 10.0 + z1;
            x2 = z2;
            y = -10.0 - 3.0 * z1 + sqrt(6.0) * z3;
        }

        return DataPoint(x1, x2, y);
    }

    void generateDataset(int N, Dataset& dataset)
    {
        for (int i = 0; i < N; i++)
        {
            dataset.addSample(generateSample());
        }
    }
};

class CubicPolynomial
{
private:
    vector<double> weights;

public:
    CubicPolynomial()
    {
        weights.assign(10, 0.0);
    }

    vector<double> getFeatures(const DataPoint& sample) const
    {
        double x1 = sample.getX1();
        double x2 = sample.getX2();

        return {
            1.0,
            x1,
            x2,
            x1 * x1,
            x1 * x2,
            x2 * x2,
            x1 * x1 * x1,
            x1 * x1 * x2,
            x1 * x2 * x2,
            x2 * x2 * x2
        };
    }

    void setWeights(const vector<double>& newWeights)
    {
        if (newWeights.size() != 10)
        {
            cerr << "Error: Expected 10 polynomial coefficients." << endl;
            return;
        }

        weights = newWeights;
    }

    double predict(const DataPoint& sample) const
    {
        vector<double> features = getFeatures(sample);

        double prediction = 0.0;

        for (int i = 0; i < 10; i++)
        {
            prediction += weights[i] * features[i];
        }

        return prediction;
    }

    void printWeights() const
    {
        cout << "Polynomial Coefficients:" << endl;

        for (int i = 0; i < 10; i++)
        {
            cout << "w" << i << ": " << weights[i] << endl;
        }
    }
};

class MatrixSolver
{
public:
    vector<double> solve(vector<vector<double>> A,
                         vector<double> b) const
    {
        int n = static_cast<int>(A.size());

        if (n == 0 || static_cast<int>(b.size()) != n)
        {
            throw invalid_argument("Invalid matrix dimensions");
        }

        for (const auto& row : A)
        {
            if (static_cast<int>(row.size()) != n)
            {
                throw invalid_argument("Matrix must be square");
            }
        }

        // Forward elimination
        for (int i = 0; i < n; i++)
        {
            // Find row with the largest pivost
            int pivotRow = i;

            for (int j = i + 1; j < n; j++)
            {
                if (abs(A[j][i]) > abs(A[pivotRow][i]))
                {
                    pivotRow = j;
                }
            }

            // Check for singular or nearly singular matrix
            if (abs(A[pivotRow][i]) < 1e-12)
            {
                throw runtime_error(
                    "Matrix is singular or nearly singular"
                );
            }

            // Swap pivot row into position
            swap(A[i], A[pivotRow]);
            swap(b[i], b[pivotRow]);

            // Eliminate entries below pivot
            for (int j = i + 1; j < n; j++)
            {
                double factor = A[j][i] / A[i][i];

                for (int k = i; k < n; k++)
                {
                    A[j][k] -= factor * A[i][k];
                }

                b[j] -= factor * b[i];
            }
        }

        // Back substitution
        vector<double> x(n, 0.0);

        for (int i = n - 1; i >= 0; i--)
        {
            double sum = b[i];

            for (int j = i + 1; j < n; j++)
            {
                sum -= A[i][j] * x[j];
            }

            x[i] = sum / A[i][i];
        }

        return x;
    }

};

class MLEstimator
{
private:
    MatrixSolver solver;

public:
    vector<double> estimate(const Dataset& trainingData) const
    {
        const int numFeatures = 10;

        CubicPolynomial polynomial;

        // Initialize Z^T Z and Z^T y
        vector<vector<double>> A(
            numFeatures,
            vector<double>(numFeatures, 0.0)
        );

        vector<double> b(numFeatures, 0.0);

        // Process every training observation
        for (int i = 0; i < trainingData.getSize(); i++)
        {
            const DataPoint& sample = trainingData.getSample(i);

            vector<double> features = polynomial.getFeatures(sample);

            double y = sample.getY();

            // Construct Z^T y
            for (int j = 0; j < numFeatures; j++)
            {
                b[j] += features[j] * y;

                // Construct Z^T Z
                for (int k = 0; k < numFeatures; k++)
                {
                    A[j][k] += features[j] * features[k];
                }
            }
        }

        // Solve (Z^T Z)w = Z^T y
        return solver.solve(A, b);
    }
};

class MAPEstimator
{
private:
    MatrixSolver solver;

public:
    vector<double> estimate(const Dataset& trainingData,
                            double gamma,
                            double noiseVariance = 1.0) const
    {
        if (gamma <= 0.0 || noiseVariance <= 0.0)
        {
            throw invalid_argument(
                "Gamma and noise variance must be positive"
            );
        }

        const int numFeatures = 10;

        CubicPolynomial polynomial;

        vector<vector<double>> A(
            numFeatures,
            vector<double>(numFeatures, 0.0)
        );

        vector<double> b(numFeatures, 0.0);

        // Construct Z^T Z and Z^T y
        for (int i = 0; i < trainingData.getSize(); i++)
        {
            const DataPoint& sample = trainingData.getSample(i);

            vector<double> features = polynomial.getFeatures(sample);

            double y = sample.getY();

            for (int j = 0; j < numFeatures; j++)
            {
                b[j] += features[j] * y;

                for (int k = 0; k < numFeatures; k++)
                {
                    A[j][k] += features[j] * features[k];
                }
            }
        }

        // Add MAP regularization to diagonal
        double lambda = noiseVariance / gamma;

        for (int j = 0; j < numFeatures; j++)
        {
            A[j][j] += lambda;
        }

        // Solve MAP system
        return solver.solve(A, b);
    }
};

double calculateMSE(const CubicPolynomial& polynomial, 
                    const Dataset& dataset)
{
    if (dataset.getSize() == 0)
    {
        throw invalid_argument("Cannot evaluate an empty dataset");
    }

    double totalSquaredError = 0.0;

    for (int i = 0; i < dataset.getSize(); i++)
    {
        const DataPoint& sample = dataset.getSample(i);

        double prediction = polynomial.predict(sample);

        double error = sample.getY() - prediction;

        totalSquaredError += error * error;
    }

    return totalSquaredError / dataset.getSize();
}


int main()
{
    // ==========================================
    // STEP 1: Generate Training and Validation Data
    // ==========================================

    GaussianMixtureGenerator generator(42);

    Dataset Dtrain;
    Dataset Dvalidate;

    generator.generateDataset(100, Dtrain);
    generator.generateDataset(1000, Dvalidate);

    cout << "========== QUESTION 2: DATA GENERATION ==========" << endl;
    cout << "Training Samples: " << Dtrain.getSize() << endl;
    cout << "Validation Samples: " << Dvalidate.getSize() << endl;

    Dtrain.saveCSV("Q2_Train.csv");
    Dvalidate.saveCSV("Q2_Validate.csv");


    // ==========================================
    // STEP 2: Maximum Likelihood Estimation
    // ==========================================

    cout << "\n========== MAXIMUM LIKELIHOOD ESTIMATION =========="
         << endl;

    MLEstimator mlEstimator;

    vector<double> mlWeights = mlEstimator.estimate(Dtrain);

    CubicPolynomial mlModel;
    mlModel.setWeights(mlWeights);

    mlModel.printWeights();

    double mlTrainingMSE = calculateMSE(mlModel, Dtrain);
    double mlValidationMSE = calculateMSE(mlModel, Dvalidate);

    cout << "\nML Training MSE: " << mlTrainingMSE << endl;
    cout << "ML Validation MSE: " << mlValidationMSE << endl;


    // ==========================================
    // STEP 3: Maximum A Posteriori Estimation
    // ==========================================

    cout << "\n========== MAXIMUM A POSTERIORI ESTIMATION =========="
         << endl;

    MAPEstimator mapEstimator;

    const double noiseVariance = 1.0;

    vector<double> gammaValues = {
        1e-9, 1e-8, 1e-7, 1e-6,
        1e-5, 1e-4, 1e-3, 1e-2,
        1e-1, 1.0, 1e1, 1e2,
        1e3, 1e4, 1e5, 1e6
    };

    ofstream resultsFile("Q2_MAP_ML_Comparison.csv");

    if (!resultsFile.is_open())
    {
        cerr << "Error: Cannot create comparison CSV." << endl;
        return 1;
    }

    resultsFile << "Gamma,MAP_Training_MSE,MAP_Validation_MSE,"
                << "ML_Validation_MSE\n";

    resultsFile << setprecision(12);

    cout << "Gamma\t\tTraining MSE\tValidation MSE" << endl;

    double bestGamma = 0.0;
    double bestValidationMSE = numeric_limits<double>::infinity();

    for (double gamma : gammaValues)
    {
        vector<double> mapWeights =
            mapEstimator.estimate(Dtrain, gamma, noiseVariance);

        CubicPolynomial mapModel;
        mapModel.setWeights(mapWeights);

        double mapTrainingMSE = calculateMSE(mapModel, Dtrain);
        double mapValidationMSE = calculateMSE(mapModel, Dvalidate);

        cout << gamma << "\t\t"
             << mapTrainingMSE << "\t\t"
             << mapValidationMSE << endl;

        resultsFile << gamma << ","
                    << mapTrainingMSE << ","
                    << mapValidationMSE << ","
                    << mlValidationMSE << "\n";

        if (mapValidationMSE < bestValidationMSE)
        {
            bestValidationMSE = mapValidationMSE;
            bestGamma = gamma;
        }
    }

    resultsFile.close();


    // ==========================================
    // STEP 4: Compare ML and MAP Performance
    // ==========================================

    double improvement =
        (mlValidationMSE - bestValidationMSE) /
        mlValidationMSE * 100.0;

    cout << "\n========== MODEL COMPARISON ==========" << endl;

    cout << "ML Validation MSE: "
         << mlValidationMSE << endl;

    cout << "Best Tested MAP Gamma: "
         << bestGamma << endl;

    cout << "Best MAP Validation MSE: "
         << bestValidationMSE << endl;

    cout << "MAP Improvement Over ML: "
         << improvement << "%" << endl;

    cout << "\nSaved: Q2_MAP_ML_Comparison.csv" << endl;

    return 0;
}
