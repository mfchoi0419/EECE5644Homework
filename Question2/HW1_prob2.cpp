#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace std;

// Dimensions, gaussian components, classes
const int DIM = 3;
const int NUM_GAUSSIANS = 4;
const int NUM_CLASSES = 3;
const int NUM_SAMPLES = 10000;

// Three-dimensional observation
using Vector3 = array<double, DIM>;

// Store observation and true class label
struct Sample
{
    Vector3 x;
    int label;
};

// Gaussian component means
const array<Vector3, NUM_GAUSSIANS> means = 
{{
    {0.0, 0.0, 0.0},
    {2.5, 0.0, 0.0},
    {1.25, 2.165, 0.0},
    {1.25, 0.722, 2.041}
}};

// Class prior probabilities
const array<double, NUM_CLASSES> priors = 
{
    0.3, 0.3, 0.4
};

// All Gaussian components have covariance I_3
const double STANDARD_DEVIATION = 1.0;

Vector3 GenerateGaussianSample(
    const Vector3& mean,
    mt19937& generator)
{
    normal_distribution<double> normal(0.0, STANDARD_DEVIATION);

    Vector3 x;

    for (int i = 0; i < DIM; i++)
    {
        x[i] = mean[i] + normal(generator);
    }

    return x;
}

vector<Sample> GenerateDataset(mt19937& generator)
{
    vector<Sample> samples;
    samples.reserve(NUM_SAMPLES);

    // Probabilities for four Gaussian components
    discrete_distribution<int> componentDistribution(
        {0.3, 0.3, 0.2, 0.2}
    );

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        // Select one of four Gaussian components
        int component = componentDistribution(generator);

        // Generate observation from selected component
        Vector3 x = GenerateGaussianSample(
            means[component], generator
        );

        // Assign corresponding true class label
        int label;

        if (component == 0)
        {
            label = 1;
        }
        else if (component == 1)
        {
            label = 2;
        }
        else
        {
            label = 3;
        }

        samples.push_back({x, label});
    }

    return samples;
}

double GaussianScore(const Vector3& x, const Vector3& mean)
{
    double squaredDistance = 0.0;

    for (int i = 0; i < DIM; i++)
    {
        double difference = x[i] - mean[i];

        squaredDistance += difference * difference;
    }

    return exp(-0.5 * squaredDistance);
}

int MAPClassifier(const Vector3& x)
{
    // Calculate three class decision scores
    double score1 = priors[0] * GaussianScore(x, means[0]);
    double score2 = priors[1] * GaussianScore(x, means[1]);
    double score3 = priors[2] * (0.5 * GaussianScore(x, means[2]) +  
                                 0.5 * GaussianScore(x, means[3])
    );

    // Select class with highest score
    if (score1 >= score2 && score1 >= score3)
    {
        return 1;
    }
    else if (score2 >= score1 && score2 >= score3)
    {
        return 2;
    }
    else
    {
        return 3;
    }
}

void EvaluateMAP(const vector<Sample>& samples)
{
    int counts[NUM_CLASSES][NUM_CLASSES] = {};
    int classTotals[NUM_CLASSES] = {};

    int correct = 0;

    // Classify every observation
    for (const Sample& sample : samples)
    {
        int predicted = MAPClassifier(sample.x);
        int actual = sample.label;

        // Convert class labels (1, 2, 3) to indices (0, 1, 2)
        counts[predicted - 1][actual - 1]++;
        classTotals[actual - 1]++;

        if (predicted == actual)
        {
            correct++;
        }
    }

    // Display confusion matrix
    cout << "\nMAP Confusion Matrix (Counts):" << endl;
    cout << "Rows = Predicted, Columns = True" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            cout << setw(8) << counts[i][j];
        }

        cout << endl;
    }

    // Display normalized confusion matrix
    cout << "\nMAP Confusion Matrix (Probabilities):" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            double probability = 0.0;

            if (classTotals[j] > 0)
            {
                probability = static_cast<double>(counts[i][j]) / classTotals[j];
            }

            cout << fixed << setprecision(4) << setw(10) << probability;
        }

        cout << endl;
    }

    // Calculate empirical probability of error
    int incorrect = static_cast<int>(samples.size()) - correct;

    double errorProbability = static_cast<double>(incorrect) / samples.size();

    cout << "\nCorrect classifications: " << correct << endl;
    cout << "Incorrect classifications: " << incorrect << endl;

    cout << "Empirical MAP error probability: "
         << fixed << setprecision(4)
         << errorProbability << endl;
}

void ExportMAPPlotData(const vector<Sample>& samples)
{
    ofstream files[NUM_CLASSES][2];

    // Separate data by true class and classification result
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            string filename = "MAP_Class" + to_string(i + 1);

            if (j == 0)
            {
                filename += "_Correct.dat";
            }
            else
            {
                filename += "_Incorrect.dat";
            }

            files[i][j].open(filename);
        }
    }

    // Classify each observation and export coordinates
    for (const Sample& sample : samples)
    {
        int predicted = MAPClassifier(sample.x);

        int classIndex = sample.label - 1;
        int resultIndex = (predicted == sample.label) ? 0 : 1;

        files[classIndex][resultIndex]
            << sample.x[0] << " "
            << sample.x[1] << " "
            << sample.x[2] << endl;
    }

    // Close output files
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            files[i][j].close();
        }
    }

     cout << "\nMAP plotting data exported successfully." << endl;
}

int ERMClassifier10(const Vector3& x)
{
    // Calculate the prior-weighted Gaussian scores
    double score1 = priors[0] * GaussianScore(x, means[0]);
    double score2 = priors[1] * GaussianScore(x, means[1]);
    double score3 = priors[2] * (
        0.5 * GaussianScore(x, means[2]) +
        0.5 * GaussianScore(x, means[3])
    );

    // Calculate conditional risks using Lambda_10
    double risk1 = score2 + 10.0 * score3;
    double risk2 = score1 + 10.0 * score3;
    double risk3 = score1 + score2;

    // Select decision with minimum conditional risk
    if (risk1<= risk2 && risk1 <= risk3)
    {
        return 1;
    }
    else if (risk2 < risk1 && risk2 <= risk3)
    {
        return 2;
    }
    else
    {
        return 3;
    }
}

void EvaluateERM10(const vector<Sample>& samples)
{
    int counts[NUM_CLASSES][NUM_CLASSES] = {};
    int classTotals[NUM_CLASSES] = {};

    // Loss matrix: rows = predicted, columns = true
    const double loss[NUM_CLASSES][NUM_CLASSES] = 
    {
        {0.0, 1.0, 10.0},
        {1.0, 0.0, 10.0},
        {1.0, 1.0, 0.0}
    };

    int correct = 0;
    double totalLoss = 0.0;

    // Classify all observations
    for (const Sample& sample : samples)
    {
        int predicted = ERMClassifier10(sample.x);
        int actual = sample.label;

        int i = predicted - 1;
        int j = actual - 1;

        counts[i][j]++;
        classTotals[j]++;

        totalLoss += loss[i][j];

        if (predicted == actual)
        {
            correct++;
        }
    }

    // Print confusion matrix counts
    cout << "\nERM Lambda_10 Confusion Matrix(Counts):" << endl;
    cout << "Rows = Predicted, Columns = True" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            cout << setw(8) << counts[i][j];
        }
        
        cout << endl;
    }

    // Print normalized confusion matrix
    cout << "\nERM Lambda_10 Confusion Matrix (Probabilities):" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            double probability = 0.0;

            if (classTotals[j] > 0)
            {
                probability = static_cast<double>(counts[i][j]) / classTotals[j];
            }

            cout << fixed << setprecision(4)
                 << setw(10) << probability;
        }

        cout << endl;
    }

    // Calculate empirical classification error
    int incorrect = static_cast<int>(samples.size()) - correct;

    double errorProbability = static_cast<double>(incorrect) / samples.size();

    // Calculate empirical expected risk
    double expectedRisk = totalLoss / samples.size();

    cout << "\nERM Lambda_10 Results:" << endl;
    cout << "Correct classifications: " << correct << endl;
    cout << "Incorrect classifications: " << incorrect << endl;
    cout << "Empirical error probability: " 
         << fixed << setprecision(4)
         << errorProbability << endl;
    cout << "Empirical expected risk (Lambda_10): "
         << fixed << setprecision(4)
         << expectedRisk << endl;
}

int ERMClassifier100(const Vector3& x)
{
    // Calculate prior-weighted Gaussian scores
    double score1 = priors[0] * GaussianScore(x, means[0]);
    double score2 = priors[1] * GaussianScore(x, means[1]);
    double score3 = priors[2] * (
        0.5 * GaussianScore(x, means[2]) +
        0.5 * GaussianScore(x, means[3])
    );

    // Calculate conditional risks using Lambda_100
    double risk1 = score2 + 100.0 * score3;
    double risk2 = score1 + 100.0 * score3;
    double risk3 = score1 + score2;

    // Select decision withn minimum conditional risk
    if (risk1 <= risk2 && risk1 <= risk3)
    {
        return 1;
    }
    else if (risk2 < risk1 && risk2 <= risk3)
    {
        return 2;
    }
    else
    {
        return 3;
    }
}

void EvaluateERM100(const vector<Sample>& samples)
{
    int counts[NUM_CLASSES][NUM_CLASSES] = {};
    int classTotals[NUM_CLASSES] = {};

    // Loss matrix: Rows = Predicted, Columns = True
    const double loss[NUM_CLASSES][NUM_CLASSES] = {
        {0.0, 1.0, 100.0},
        {1.0, 0.0, 100.0},
        {1.0, 1.0, 0.0}
    };

    int correct = 0;
    double totalLoss = 0.0;

    // Classify all observations
    for (const Sample& sample : samples)
    {
        int predicted = ERMClassifier100(sample.x);
        int actual = sample.label;

        int i = predicted - 1;
        int j = actual - 1;

        counts[i][j]++;
        classTotals[j]++;

        totalLoss += loss[i][j];

        if (predicted == actual)
        {
            correct++;
        }
    }

    // Print confusion matrix counts
    cout << "\nERM Lambda_100 Confusion Matrix (Counts):" << endl;
    cout << "Rows = Predicted, Columns = True" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            cout << setw(8) << counts[i][j];
        }

        cout << endl;
    }

    // Print normalized confusion matrix
    cout << "\nERM Lambda_100 Confusion Matrix (Probabilities):" << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            double probability = 0.0;

            if (classTotals[j] > 0)
            {
                probability = static_cast<double>(counts[i][j]) / classTotals[j];
            }

            cout << fixed << setprecision(4)
                 << setw(10) << probability;
        }

        cout << endl;
    }

    // Calculate empirical classification error
    int incorrect = static_cast<int>(samples.size()) - correct;

    double errorProbability = static_cast<double>(incorrect) / samples.size();

    // Calculate empirical expected risk
    double expectedRisk = totalLoss / samples.size();

    cout << "\nERM Lambda_100 Results:" << endl;
    cout << "Correct classifications: " << correct << endl;
    cout << "Incorrect classifications: " << incorrect << endl;

    cout << "Empirical error probability: "
         << fixed << setprecision(4)
         << errorProbability << endl;

    cout << "Empirical expected risk (Lambda_100): "
         << fixed << setprecision(4)
         << expectedRisk << endl;
}

void ExportERMPlotData(const vector<Sample>& samples)
{
    ofstream files10[NUM_CLASSES][2];
    ofstream files100[NUM_CLASSES][2];

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        string classNumber = to_string(i + 1);

        files10[i][0].open(
            "ERM10_Class" + classNumber + "_Correct.dat"
        );
        files10[i][1].open(
            "ERM10_Class" + classNumber + "_Incorrect.dat"
        );
        files100[i][0].open(
            "ERM100_Class" + classNumber + "_Correct.dat"
        );
        files100[i][1].open(
            "ERM100_Class" + classNumber + "_Incorrect.dat"
        );
    }

    for (const Sample& sample : samples)
    {
        int trueClassIndex = sample.label - 1;
        
        int prediction10 = ERMClassifier10(sample.x);
        int prediction100 = ERMClassifier100(sample.x);

        int result10 = (prediction10 == sample.label) ? 0 : 1;
        int result100 = (prediction100 == sample.label) ? 0 : 1;

        files10[trueClassIndex][result10]
            << sample.x[0] << " "
            << sample.x[1] << " "
            << sample.x[2] << "\n";
        
        files100[trueClassIndex][result100]
            << sample.x[0] << " "
            << sample.x[1] << " "
            << sample.x[2] << "\n";
    }

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            files10[i][j].close();
            files100[i][j].close();
        }
    }

    cout << "\nERM plotting data exported successfully." << endl;
}


int main()
{
    // Generate 10,000 labeled observations
    mt19937 generator(42);

    vector<Sample> samples = GenerateDataset(generator);

    // Count observations in each class
    int classCounts[NUM_CLASSES] = {0};

    for (const Sample& sample : samples)
    {
        classCounts[sample.label - 1]++;
    }

    // Display dataset summary
    cout << "Question 2: Gaussian Mixture Classification"
         << endl;

    cout << "Total observations: "
         << samples.size() << endl;

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        cout << "Class " << i + 1
             << ": " << classCounts[i]
             << " observations" << endl;
    }

    // Evaluate MAP classifier
    cout << "\nMAP Classification Results:" << endl;
    EvaluateMAP(samples);

    // Evaluate ERM classifier with Lambda_10
    cout << "\nERM Lambda_10 Classification Results:"
         << endl;
    EvaluateERM10(samples);

    // Evaluate ERM classifier with Lambda_100
    cout << "\nERM Lambda_100 Classification Results:"
         << endl;
    EvaluateERM100(samples);

    // Export visualization data
    ExportMAPPlotData(samples);
    ExportERMPlotData(samples);

    return 0;
}
