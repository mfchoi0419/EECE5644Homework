#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// Sample size and dimension
const int N = 10000;
const int DIM = 4;

// ========================================
// SHARED TYPES AND PARAMETERS
// ========================================
struct Sample
{
    array<double, DIM> features;
    int label;
};
struct Evaluation
{
    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;
    double TPR = 0.0;
    double FPR = 0.0;
    double FNR = 0.0;
    double error = 0.0;
};
// Define Gaussian parameters
using Vector4 = array<double, DIM>;
using Matrix4 = array<Vector4, DIM>;
Vector4 m0 = {-0.5, -0.5, -0.5, -0.5};
Vector4 m1 = {1.0, 1.0, 1.0, 1.0};
Matrix4 C0 = {{
    {{ 2.0, -0.5, 0.3, 0.0}},
    {{ -0.5, 1.0, -0.5, 0.0}},
    {{ 0.3, -0.5, 1.0, 0.0}},
    {{ 0.0, 0.0, 0.0, 2.0}},
}};
Matrix4 C1 = {{
    {{ 1.0, 0.3, -0.2, 0.0}},
    {{ 0.3, 2.0, 0.3, 0.0}},
    {{ -0.2, 0.3, 1.0, 0.0}},
    {{ 0.0, 0.0, 0.0, 3.0}},
}};

// ========================================
// SHARED DATA GENERATION AND MATH HELPERS
// ========================================

Matrix4 Cholesky(const Matrix4& C)
{
    Matrix4 A{};
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = C[i][j];
            for (int k = 0; k < j; k++)
            {
                sum -= A[i][k] * A[j][k];
            }
            if (i == j)
            {
                if (sum <= 0.0)
                {
                    cerr << "Cholesky error at index " << i << ": " << sum << endl;
                    return Matrix4{};
                }
                A[i][j] = sqrt(sum);
            }
            else
            {
                A[i][j] = sum / A[j][j];
            }
        }
    }
    return A;
}

Vector4 GenerateGaussian(const Vector4& mean, const Matrix4& A, mt19937& gen)
{
    normal_distribution<double> normal(0.0, 1.0);
    Vector4 Z{};
    Vector4 X{};
    for (int i = 0; i < DIM; i++)
    {
        Z[i] = normal(gen);
    }
    for (int i = 0; i < DIM; i++)
    {
        X[i] = mean[i];
        for (int j = 0; j < DIM; j++)
        {
            X[i] += A[i][j] * Z[j];
        }
    }
    return X;
}

vector<Sample> GenerateDataset(
    const Vector4& m0,
    const Vector4& m1,
    const Matrix4& A0,
    const Matrix4& A1,
    mt19937& gen)
{
    vector<Sample> dataset;
    dataset.reserve(N);
    bernoulli_distribution classDist(0.65);
    for (int i = 0; i < N; i++)
    {
        Sample sample;
        sample.label = classDist(gen) ? 1 : 0;
        if (sample.label == 0)
        {
            sample.features = GenerateGaussian(m0, A0, gen);
        }
        else
        {
            sample.features = GenerateGaussian(m1, A1, gen);
        }
        dataset.push_back(sample);
    }
    return dataset;
}

void SaveDataset(const vector<Sample>& dataset, const string& filename)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    file << setprecision(17);
    // Column headers
    file << "x1,x2,x3,x4,label\n";
    // Save each observation
    for (const Sample& sample : dataset)
    {
        for (int i = 0; i < DIM; i++)
        {
            file << sample.features[i] << ",";
        }
        file << sample.label << "\n";
    }
    file.close();
    cout << "Dataset saved to " << filename << endl;
}

double GaussianLogLikelihood(
    const Vector4& x,
    const Vector4& mean,
    const Matrix4& A)
{
    Vector4 y{};
    double logDet = 0.0;
    double quadratic = 0.0;
    // Solve A * y = x - mean
    for (int i = 0; i < DIM; i++)
    {
        double sum = x[i] - mean[i];
        for (int j = 0; j < i; j++)
        {
            sum -= A[i][j] * y[j];
        }
        y[i] = sum / A[i][i];
        // Log determinant
        logDet += 2.0 * log(A[i][i]);
    }
    // Compute y^T * y
    for (int i = 0; i < DIM; i++)
    {
        quadratic += y[i] * y[i];
    }
    const double PI = acos(-1.0);
    return -0.5 * ( DIM * log(2.0 * PI) + logDet + quadratic );
}

// ========================================
// QUESTION 1 - PART A: FULL GAUSSIAN
// ========================================

int LiklihoodRatioClassifier(
    const Vector4& x,
    const Vector4& m0,
    const Vector4& m1,
    const Matrix4& A0,
    const Matrix4& A1,
    double gamma)
{
    double logP0 = GaussianLogLikelihood(x, m0, A0);
    double logP1 = GaussianLogLikelihood(x, m1, A1);
    double logLR = logP1 - logP0;
    if (logLR > log(gamma))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

Evaluation EvaluateClassifier(
    const vector<Sample>& dataset,
    const Vector4& m0,
    const Vector4& m1,
    const Matrix4& A0,
    const Matrix4& A1,
    double gamma)
{
    Evaluation result;
    // Classify every observation
    for (const Sample& sample : dataset)
    {
        int prediction = LiklihoodRatioClassifier(
            sample.features, m0, m1, A0, A1, gamma
        );
        if (sample.label == 1 && prediction == 1)
        {
            result.TP++;
        }
        else if (sample.label == 0 && prediction == 0)
        {
            result.TN++;
        }
        else if (sample.label == 0 && prediction == 1)
        {
            result.FP++;
        }
        else if (sample.label == 1 && prediction == 0)
        {
            result.FN++;
        }
    }
    // Calculate evaluation metrics
    result.TPR = static_cast<double>(result.TP) / (result.TP + result.FN);
    result.FPR = static_cast<double>(result.FP) / (result.FP + result.TN);
    result.FNR = static_cast<double>(result.FN) / (result.FN + result.TP);
    result.error = static_cast<double>(result.FP + result.FN) / dataset.size();
    return result;
}

void GenerateROC(
    const vector<Sample>& dataset,
    const Vector4& m0,
    const Vector4& m1,
    const Matrix4& A0,
    const Matrix4& A1,
    const string& filename,
    const string& pointsFilename)
{
    // Store each observation's log-likelihood ratio and label
    vector<pair<double, int>> scores;

    for (const Sample& sample : dataset)
    {
        double logP0 = GaussianLogLikelihood(
            sample.features, m0, A0
        );
        double logP1 = GaussianLogLikelihood(
            sample.features, m1, A1
        );
        double logLR = logP1 - logP0;
        scores.push_back({logLR, sample.label});
    }

    // Sort from largest to smallest for log-likelihood
    sort(scores.begin(), scores.end(),
    [](const pair<double, int> &a, const pair<double, int>& b)
        {
            return a.first > b.first;
        });

    int total0 = 0;
    int total1 = 0;

    for (const auto& score : scores)
    {
        if (score.second == 0)
        {
            total0++;
        }
        else
        {
            total1++;
        }
    }

    // Every initial observation is classified as 0
    int TP = 0;
    int FP = 0;
    int TN = total0;
    int FN = total1;

    // Track empirical minimum-error operating point
    double minError = static_cast<double>(FN) / dataset.size();
    double bestLogGamma = 0.0;
    double bestTPR = 0.0;
    double bestFPR = 0.0;
    double bestFNR = 1.0;
    size_t bestIndex = 0;
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    file << setprecision(17);
    file << "log_gamma,FPR,TPR,FNR,error\n";

    // Initial ROC point: (0,0)
    file << "inf,0,0,1," << static_cast<double>(FN) / dataset.size() << "\n";

    // Sweep threshold from high to low
    size_t i = 0;

    while (i < scores.size())
    {
        double currentScore = scores[i].first;

        // Process all observations with same score
        while (i < scores.size() && scores[i].first == currentScore)
        {
            if (scores[i].second == 1)
            {
                TP++;
                FN--;
            }
            else
            {
                FP++;
                TN--;
            }
            i++;
        }

        double TPR = static_cast<double>(TP) / total1;
        double FPR = static_cast<double>(FP) / total0;
        double FNR = static_cast<double>(FN) / total1;
        double error = static_cast<double>(FP + FN) / dataset.size();

        if (error < minError)
        {
            minError = error;
            bestLogGamma = currentScore;
            bestTPR = TPR;
            bestFPR = FPR;
            bestFNR = FNR;
            bestIndex = i;
        }

        // Point representing threshold just below
        // currentScore, so tied scores predict class 1
        file << currentScore << "," << FPR
        << "," << TPR << "," << FNR << ","
        << error << "\n";
    }

    double optimalLogGamma;

    if (bestIndex == 0)
    {
        // Threshold above largest observed score
        optimalLogGamma = scores.front().first + 1.0;
    }
    else if (bestIndex == scores.size())
    {
        // Threshold below smallest observed score
        optimalLogGamma = scores.back().first - 1.0;
    }
    else
    {
        // Midpoint between adjacent scores
        optimalLogGamma = bestLogGamma + (scores[bestIndex].first - bestLogGamma) / 2.0;
    }

    double optimalGamma = exp(optimalLogGamma);

    cout << "\nEmpirically Optimal Threshold:" << endl;
    cout << "Log gamma: " << optimalLogGamma << endl;
    cout << "Gamma: " << optimalGamma << endl;
    cout << "TPR: " << bestTPR << endl;
    cout << "FPR: " << bestFPR << endl;
    cout << "FNR: " << bestFNR << endl;
    cout << "Minimum empirical error: " << minError << endl;
    file.close();
    cout << "ROC data saved to " << filename << endl;

    ofstream points(pointsFilename);

    if (!points.is_open())
    {
        cerr << "Error creating operating points CSV." << endl;
    }
    else
    {
        points << setprecision(17);
        points << "FPR,TPR,Label\n";
        // Theoretical minimum-error threshold
        Evaluation theoretical = EvaluateClassifier(
            dataset, m0, m1, A0, A1, 0.35 / 0.65
        );
        points << theoretical.FPR << "," << theoretical.TPR
               << ",Theoretical\n";
        // Empirical minimum-error threshold
        points << bestFPR << "," << bestTPR << ",Empirical\n";
        points.close();
    }
}

void PlotROC(const string& csvFilename,
             const string& pointsFilename,
             const string& scriptFilename,
             const string& imageFilename,
             const string& plotTitle,
             const string& curveLabel)
{
    ofstream file(scriptFilename);
    if (!file.is_open())
    {
        cerr << "Error creating Gnuplot script." << endl;
        return;
    }

    // Configure image output
    file << "set terminal pngcairo size 1000, 750\n";
    file << "set output '" << imageFilename << "'\n";

    // Plot title and axis labels
    file << "set title '" << plotTitle << "'\n";
    file << "set xlabel 'False Positive Rate (FPR)'\n";
    file << "set ylabel 'True Positive Rate (TPR)'\n";

    // Set axis ranges
    file << "set xrange [0:1]\n";
    file << "set yrange [0:1]\n";

    // Grid, key, separator
    file << "set grid\n";
    file << "set key outside right\n";
    file << "set datafile separator ','\n";

    // Plot Gaussian ROC curve
    file << "plot '" << csvFilename
         << "' using 2:3 with lines linewidth 2 "
         << "title '" << curveLabel << "', \\\n";

    // Plot random classifier diagonal
    file << "x with lines dashtype 2 "
         << "title 'Random Classifier', \\\n";

    // Plot theoretical minimum-error threshold
    file << "'" << pointsFilename << "' "
         << "every ::1::1 using 1:2 "
         << "with points pointtype 7 pointsize 2 "
         << "title 'Theoretical Threshold', \\\n";

    // Plot empirically optimal threshold
    file << "'" << pointsFilename << "' "
         << "every ::2::2 using 1:2 "
         << "with points pointtype 5 pointsize 2 "
         << "title 'Empirical Threshold'\n";

    file.close();
    cout << "Gnuplot script created." << endl;
}

// Part A execution

void RunPartA(const vector<Sample>& dataset,
              const Matrix4& A0, const Matrix4& A1)
{
    // Step 8. Compute Gaussian log-likelihood test
    cout << "\nGaussian log-likelihood:" << endl;
    
    const Vector4& x = dataset[0].features;
    double logP0 = GaussianLogLikelihood(x, m0, A0);
    double logP1 = GaussianLogLikelihood(x, m1, A1);
    
    cout << "True label: " << dataset[0].label << endl;
    cout << "Log p(x|0): " << logP0 << endl;
    cout << "Log p(x|1): " << logP1 << endl;
    cout << "Log likelihood ratio: " << logP1 - logP0 << endl;
    
    // Step 9. Compute likelihood ratio test
    double gamma = 0.35 / 0.65;
    int prediction = LiklihoodRatioClassifier(
        dataset[0].features, m0, m1, A0, A1, gamma
    );
    cout << "Predicted label: " << prediction << endl;
    cout << "Actual label: " << dataset[0].label << endl;
    // Step 10. Evaluate the classifier
    Evaluation result = EvaluateClassifier(
        dataset, m0, m1, A0, A1, gamma
    );
    cout << "\nMinimum Probability of Error Classifier:" << endl;
    
    cout << "Threshold gamma: " << gamma << endl;
    cout << "\nConfusion Matrix Counts:" << endl;
    cout << "TP: " << result.TP << endl;
    cout << "TN: " << result.TN << endl;
    cout << "FP: " << result.FP << endl;
    
    cout << "FN: " << result.FN << endl;
    cout << "\nPerformance Metrics:" << endl;
    cout << "TPR: " << result.TPR << endl;
    cout << "FPR: " << result.FPR << endl;
    cout << "FNR: " << result.FNR << endl;
    cout << "Error Rate: " << result.error << endl;
    
    // Step 11: Generate ROC data for question 1, Part A
    GenerateROC(dataset, m0, m1, A0, A1, "Question1_ROC.csv", "Question1_OperatingPoints.csv");
    PlotROC("Question1_ROC.csv",
            "Question1_OperatingPoints.csv",
            "Question1_ROC_Plot.gp",
            "Question1_ROC.png",
            "Question1 Part A: ROC Curve",
            "Gaussian ROC");
}


// ========================================
// QUESTION 1 - PART B: NAIVE BAYES
// ========================================
Matrix4 DiagonalCovariance(const Matrix4& C)
{
    Matrix4 diagonal{};

    for (int i = 0; i < DIM; i++)
    {
        diagonal[i][i] = C[i][i];
    }

    return diagonal;
}
// Part B execution

void RunPartB(const vector<Sample>& dataset,
              const Matrix4& A0, const Matrix4& A1)
{
    cout << "\nQuestion 1 - Part B: Naive Bayes" << endl;
    
    // Create diagonal covariance matrices
    Matrix4 C0_NB = DiagonalCovariance(C0);
    Matrix4 C1_NB = DiagonalCovariance(C1);

    // Compute Cholesky decomposition
    Matrix4 A0_NB = Cholesky(C0_NB);
    Matrix4 A1_NB = Cholesky(C1_NB);

    // Theoretical minimum-error threshold
    double gamma = 0.35 / 0.65;

    // Evaluate Naive Bayes classifier
    Evaluation result = EvaluateClassifier(dataset, m0, m1, A0_NB, A1_NB, gamma);

    cout << "Threshold gamma: " << gamma << endl;

    cout << "\nConfusion Matrix Counts:" << endl;
    cout << "TP: " << result.TP << endl;
    cout << "TN: " << result.TN << endl;
    cout << "FP: " << result.FP << endl;
    cout << "FN: " << result.FN << endl;

    cout << "\nPerformance Metrics:" << endl;
    cout << "TPR: " << result.TPR << endl;
    cout << "FPR: " << result.FPR << endl;
    cout << "FNR: " << result.FNR << endl;
    cout << "Error Rate: " << result.error << endl;

    GenerateROC(dataset, m0, m1, A0_NB, A1_NB, "Question1_NB_ROC.csv", "Question1_NB_OperatingPoints.csv");
    PlotROC("Question1_NB_ROC.csv",
            "Question1_NB_OperatingPoints.csv",
            "Question1_NB_ROC_Plot.gp",
            "Question1_NB_ROC.png",
            "Question1 Part B: ROC Curve",
            "Naive Bayes ROC");
}


// ========================================
// QUESTION 1 - PART C: FISHER LDA
// ========================================

Vector4 EstimateClassMean(const vector<Sample>& dataset, int classLabel)
{
    Vector4 mean{};
    int count = 0;

    for (const Sample& sample : dataset)
    {
        if (sample.label == classLabel)
        {
            for (int i = 0; i < DIM; i++)
            {
                mean[i] += sample.features[i];
            }

            count++;
        }
    }

    for (int i = 0; i < DIM; i++)
    {
        mean[i] /= count;
    }

    return mean;
}

Matrix4 EstimateClassCovariance(const vector<Sample>& dataset, int classLabel, const Vector4& mean)
{
    Matrix4 covariance{};
    int count = 0;

    for (const Sample& sample : dataset)
    {
        if (sample.label == classLabel)
        {
            for (int i = 0; i < DIM; i++)
            {
                for (int j = 0; j < DIM; j++)
                {
                    covariance[i][j] +=
                        (sample.features[i] - mean[i]) *
                        (sample.features[j] - mean[j]);
                }
            }

            count++;
        }
    }

    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            covariance[i][j] /= count;
        }
    }

    return covariance;
}

Matrix4 ComputeWithinScatter(const Matrix4& C0_hat, const Matrix4& C1_hat)
{
    Matrix4 SW{};

    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            SW[i][j] = 0.5 * (C0_hat[i][j] + C1_hat[i][j]);
        }
    }

    return SW;
}

Matrix4 ComputeBetweenScatter(const Vector4& m0_hat, const Vector4& m1_hat)
{
    Matrix4 SB{};
    Vector4 difference{};

    for (int i = 0; i < DIM; i++)
    {
        difference[i] = m1_hat[i] - m0_hat[i];
    }

    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            SB[i][j] = 0.25 * difference[i] * difference[j];
        }
    }

    return SB;
}

Vector4 ComputeLDAWeights(const Matrix4& SW, const Vector4& m0_hat, const Vector4& m1_hat)
{
    Matrix4 A = SW;
    Vector4 b{};
    Vector4 w{};

    // COmpute difference between class means
    for (int i = 0; i < DIM; i++)
    {
        b[i] = m1_hat[i] - m0_hat[i];
    }

    // Gaussian elimination with partial pivoting
    for (int i = 0; i < DIM; i++)
    {
        int pivot = i;

        for (int j = i + 1; j < DIM; j++)
        {
            if (abs(A[j][i]) > abs(A[pivot][i]))
            {
                pivot = j;
            }
        }

        if (pivot != i)
        {
            swap(A[i], A[pivot]);
            swap(b[i], b[pivot]);
        }

        for (int j = i + 1; j < DIM; j++)
        {
            double factor = A[j][i] / A[i][i];

            for (int k = i; k < DIM; k++)
            {
                A[j][k] -= factor * A[i][k];
            }

            b[j] -= factor * b[i];
        }
    }

    // Back sub
    for (int i = DIM - 1; i >= 0; i--)
    {
        double sum = b[i];

        for (int j = i + 1; j < DIM; j++)
        {
            sum -= A[i][j] * w[j];
        }

        w[i] = sum / A[i][i];
    }

    return w;
}

double ProjectLDA(const Vector4& x, const Vector4& w)
{
    double projection = 0.0;

    for (int i = 0; i < DIM; i++)
    {
        projection += w[i] * x[i];
    }

    return projection;
}

int LDAClassifier(const Vector4& x, const Vector4& w, double tau)
{
    double projection = ProjectLDA(x, w);

    if (projection > tau)
    {
        return 1;
    }

    return 0;
}

Evaluation EvaluateLDA(const vector<Sample>& dataset, const Vector4& w, double tau)
{
    Evaluation result;

    for (const Sample& sample : dataset)
    {
        int prediction = LDAClassifier(sample.features, w, tau);

        if (prediction == 1 && sample.label == 1)
        {
            result.TP++;
        }
        else if (prediction == 0 && sample.label == 0)
        {
            result.TN++;
        }
        else if (prediction == 1 && sample.label == 0)
        {
            result.FP++;
        }
        else
        {
            result.FN++;
        }
    }

    result.TPR = static_cast<double>(result.TP) / (result.TP + result.FN);
    result.FPR = static_cast<double>(result.FP) / (result.FP + result.TN);
    result.FNR = static_cast<double>(result.FN) / (result.TP + result.FN);
    result.error = static_cast<double>(result.FP + result.FN) / dataset.size();

    return result;
}

void GenerateLDAROC(const vector<Sample>& dataset, const Vector4& w, const string& filename)
{
    vector<pair<double, int>> scores;

    int totalClass0 = 0;
    int totalClass1 = 0;

    for (const Sample& sample : dataset)
    {
        double projection = ProjectLDA(sample.features, w);
        scores.push_back({projection, sample.label});

        if (sample.label == 0)
        {
            totalClass0++;
        }
        else
        {
            totalClass1++;
        }
    }

    sort(scores.begin(), scores.end(), 
        [](const pair<double, int>& a, const pair<double, int>& b)
        {
            return a.first > b.first;
        });
    
    ofstream file(filename);
    file << "tau,FPR,TPR,FNR,error\n";
    file << setprecision(12);

    int TP = 0;
    int FP = 0;
    int FN = totalClass1;

    double bestError = static_cast<double>(FN) / dataset.size();
    double bestTau = scores.front().first + 1.0;
    double bestTPR = 0.0;
    double bestFPR = 0.0;
    double bestFNR = 1.0;

    file << bestTau << ",0,0,1," << bestError << "\n";

    int i = 0;

    while (i < static_cast<int>(scores.size()))
    {
        double currentScore = scores[i].first;

        while (i < static_cast<int>(scores.size()) && scores[i].first == currentScore)
        {
            if (scores[i].second == 1)
            {
                TP++;
                FN--;
            }
            else
            {
                FP++;
            }

                i++;
            }

        double TPR = static_cast<double>(TP) / totalClass1;
        double FPR = static_cast<double>(FP) / totalClass0;
        double FNR = static_cast<double>(FN) / totalClass1;
        double error = static_cast<double>(FP + FN) / dataset.size();

        // Select a threshold between consectutive projection scores
        double tau = currentScore;

        if (i < static_cast<int>(scores.size()))
        {
            tau = 0.5 * (currentScore + scores[i].first);
        }
        else
        {
            tau = currentScore - 1.0;
        }

        file << tau << "," << FPR << "," << TPR <<
        "," << FNR << "," << error << "\n";

        if (error < bestError)
        {
            bestError = error;
            bestTau = tau;
            bestTPR = TPR;
            bestFPR = FPR;
            bestFNR = FNR;
        }
    }

    file.close();

    cout << "\nFisher LDA Empirically Optimal Threshold:" << endl;
    cout << "Tau: " << bestTau << endl;
    cout << "TPR: " << bestTPR << endl;
    cout << "FPR: " << bestFPR << endl;
    cout << "FNR: " << bestFNR << endl;
    cout << "Minimum empirical error: " << bestError << endl;
    cout << "ROC data saved to " << filename << endl;
}

void PlotLDAROC(const string& csvFilename, 
                const string& scriptFilename, 
                const string& imageFilename)
{
    ofstream script(scriptFilename);

    script << "set terminal pngcairo size 1000,750 enhanced font 'Arial,12'\n";
    script << "set output '" << imageFilename << "'\n";
    script << "set title 'Fisher LDA ROC Curve'\n";
    script << "set xlabel 'False Positive Rate (FPR)'\n";
    script << "set ylabel 'True Positive Rate (TPR)'\n";
    script << "set xrange [0:1]\n";
    script << "set yrange [0:1]\n";
    script << "set grid\n";
    script << "set key bottom right\n";
    script << "set datafile separator ','\n";

    script << "set object 1 circle at 0.0388043,0.94142 "
           << "size screen 0.007 fillstyle solid "
           << "fillcolor 'red' front\n";

    script << "plot '" << csvFilename 
           << "' using 2:3 with lines linewidth 2 "
           << "title 'Fisher LDA ROC', "
           << "x with lines dashtype 2 "
           << "title 'Random Classifier'\n";

    script.close();

    cout << "Fisher LDA Gnuplot script created." << endl;
}

// Part C execution

void RunPartC(const vector<Sample>& dataset)
{
    cout << "\nQuestion 1 - Part C: Fisher LDA" << endl;

    // Estimate sample means
    Vector4 m0_hat = EstimateClassMean(dataset, 0);
    Vector4 m1_hat = EstimateClassMean(dataset, 1);

    // Estimate sample covariance matrices
    Matrix4 C0_hat = EstimateClassCovariance(dataset, 0, m0_hat);
    Matrix4 C1_hat = EstimateClassCovariance(dataset, 1, m1_hat);

    // Print class 0 covariance
    cout << "\nEstimated Class 0 covariance:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << C0_hat[i][j] << " ";
        }
        cout << endl;
    }

    // Print class 1 covariance
    cout << "\nEstimated Class 1 covariance:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << C1_hat[i][j] << " ";
        }
        cout << endl;
    }

    // Print estimated means
    cout << "\nEstimated Class 0 mean:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        cout << m0_hat[i] << " ";
    }
    cout << endl;

    cout << "\nEstimated Class 1 mean:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        cout << m1_hat[i] << " ";
    }
    cout << endl;

    // Compute scatter matrices
    Matrix4 SW = ComputeWithinScatter(C0_hat, C1_hat);
    Matrix4 SB = ComputeBetweenScatter(m0_hat, m1_hat);

    // Print within-class scatter matrix
    cout << "\nWithin-class scatter matrix:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << SW[i][j] << " ";
        }
        cout << endl;
    }

    // Print between-class scatter matrix
    cout << "\nBetween-class scatter matrix:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            cout << SB[i][j] << " ";
        }
        cout << endl;
    }

    // Compute Fisher LDA projection weights
    Vector4 w = ComputeLDAWeights(SW, m0_hat, m1_hat);

    // Print projection weights
    cout << "\nFisher LDA Projection Vector:" << endl;
    for (int i = 0; i < DIM; i++)
    {
        cout << w[i] << " ";
    }
    cout << endl;

    // Test Fisher LDA projections
    cout << "\nFirst 5 Fisher LDA projections:" << endl;
    for (int i = 0; i < 5; i++)
    {
        double projection = ProjectLDA(dataset[i].features, w);
        cout << "Label: " << dataset[i].label 
        << " | Projection: " << projection << endl;
    }

    // Test Fisher LDA classifier at tau = 0
    double tau = 0.0;
    Evaluation ldaResult = EvaluateLDA(dataset, w, tau);

    cout << "\nFisher LDA Evaluation (tau = 0):" << endl;
    cout << "Threshold tau: " << tau << endl;

    cout << "\nConfusion Matrix Counts:" << endl;
    cout << "TP: " << ldaResult.TP << endl;
    cout << "TN: " << ldaResult.TN << endl;
    cout << "FP: " << ldaResult.FP << endl;
    cout << "FN: " << ldaResult.FN << endl;

    cout << "\nPerformance Metrics:" << endl;
    cout << "TPR: " << ldaResult.TPR << endl;
    cout << "FPR: " << ldaResult.FPR << endl;
    cout << "FNR: " << ldaResult.FNR << endl;
    cout << "Error Rate: " << ldaResult.error << endl;

    // Generate Fisher LDA ROC data and plot
    GenerateLDAROC(dataset, w, "Question1_LDA_ROC.csv");
    PlotLDAROC("Question1_LDA_ROC.csv", "Question1_LDA_ROC.gp", "Question1_LDA_ROC.png");
}

// ========================================
// MAIN
// ========================================


int main()
{
    // Scale class 0 covariance matrix
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            C0[i][j] *= 0.25;
        }
    }

    // Compute Cholesky decompositions
    Matrix4 A0 = Cholesky(C0);
    Matrix4 A1 = Cholesky(C1);

    // Generate 10,000 labeled observations
    mt19937 gen(42);

    vector<Sample> dataset =
        GenerateDataset(m0, m1, A0, A1, gen);

    // Save generated dataset
    SaveDataset(dataset, "Question1_Dataset.csv");

    // Display dataset summary
    int count0 = 0;
    int count1 = 0;

    for (const Sample& sample : dataset)
    {
        if (sample.label == 0)
        {
            count0++;
        }
        else
        {
            count1++;
        }
    }

    cout << "\nQuestion 1: Gaussian Classification" << endl;
    cout << "Total observations: " << dataset.size() << endl;
    cout << "Class 0: " << count0 << endl;
    cout << "Class 1: " << count1 << endl;

    // Evaluate all three parts
    RunPartA(dataset, A0, A1);
    RunPartB(dataset, A0, A1);
    RunPartC(dataset);

    return 0;
}
