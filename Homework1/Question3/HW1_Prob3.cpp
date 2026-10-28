#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <map>
#include <cstdio>
using namespace std;

using FeatureVector = vector<double>;
using CovarianceMatrix = vector<vector<double>>;


struct Sample
{
    FeatureVector features;
    int label;
};

struct Dataset
{
    vector<Sample> samples;
    int numFeatures = 0;
};

struct GaussianClass
{
    int label;
    int count;
    double prior;
    double lambda;

    FeatureVector mean;
    CovarianceMatrix covariance;

    CovarianceMatrix lower;
    double logDeterminant = 0.0;
};

vector<GaussianClass> models;

Dataset LoadWineDataset(const string& filename)
{
    Dataset dataset;

    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return dataset;
    }

    string line;

    // Skip CSV header
    getline(file, line);

    while (getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        stringstream ss(line);
        string value;

        Sample sample;

        // Read features separated by semicolons
        while (getline(ss, value, ';'))
        {
            sample.features.push_back(stod(value));
        }

        // Final column = quality label
        if (sample.features.empty())
        {
            continue;
        }

        sample.label = static_cast<int>(sample.features.back());

        sample.features.pop_back();

        if (dataset.numFeatures == 0)
        {
            dataset.numFeatures = static_cast<int>(sample.features.size());
        }

        if (static_cast<int>(sample.features.size()) != dataset.numFeatures)
        {
            cerr << "Skipping observation with incorrect "
                 << "feature count." << endl;
            continue;
        }

        dataset.samples.push_back(sample);
    }

    file.close();

    return dataset;
}

bool LoadHARPartition(
    const string& featureFile,
    const string& labelFile,
    Dataset& dataset)
{
    ifstream features(featureFile);
    ifstream labels(labelFile);

    if (!features.is_open() || !labels.is_open())
    {
        cerr << "Error opening HAR files: "
             << featureFile << " or "
             << labelFile << endl;
        
        return false;
    }

    string featureLine;
    string labelLine;

    while (getline(features, featureLine))
    {
        if (!getline(labels, labelLine))
        {
            cerr << "Error: Missing HAR label." << endl;
            return false;
        }

        Sample sample;
        stringstream ss(featureLine);
        double value;

        while (ss >> value)
        {
            sample.features.push_back(value);
        }

        stringstream labelStream(labelLine);

        if (!(labelStream >> sample.label))
        {
            cerr << "Error: Invalid HAR label." << endl;
            return false;
        }

        if (dataset.numFeatures == 0)
        {
            dataset.numFeatures = static_cast<int>(sample.features.size());
        }

        if (static_cast<int>(sample.features.size()) != dataset.numFeatures)
        {
            cerr << "Error: Incorrect HAR feature count."<< endl;
            return false;
        }

        dataset.samples.push_back(sample);
    }

    if (getline(labels, labelLine))
    {
        cerr << "Error: Extra HAR labels." << endl;
        return false;
    }

    return true;
}

Dataset LoadHARDataset(const string& folder)
{
    Dataset dataset;

    bool trainLoaded = LoadHARPartition(
        folder + "/train/X_train.txt",
        folder + "/train/y_train.txt",
        dataset
    );

    if (!trainLoaded)
    {
        return Dataset{};
    }

    bool testLoaded = LoadHARPartition(
        folder + "/test/X_test.txt",
        folder + "/test/y_test.txt",
        dataset
    );

    if (!testLoaded)
    {
        return Dataset{};
    }

    return dataset;
}

void StandardizeFeatures(Dataset& dataset)
{
    int N = static_cast<int>(dataset.samples.size());
    int numFeatures = dataset.numFeatures;

    if (N == 0 || numFeatures == 0)
    {
        return;
    }

    FeatureVector globalMeans(numFeatures, 0.0);
    FeatureVector globalVariances(numFeatures, 0.0);

    // Calculate global feature means
    for (const Sample& sample : dataset.samples)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            globalMeans[j] += sample.features[j];
        }
    }

    for (int j = 0; j < numFeatures; j++)
    {
        globalMeans[j] /= N;
    }

    // Calculate global feature variances
    for (const Sample& sample : dataset.samples)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            double difference =
                sample.features[j] - globalMeans[j];

            globalVariances[j] += difference * difference;
        }
    }

    for (int j = 0; j < numFeatures; j++)
    {
        globalVariances[j] /= N;
    }

    // Standardize observations
    for (Sample& sample : dataset.samples)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            double standardDeviation =
                sqrt(globalVariances[j]);

            if (standardDeviation > 0.0)
            {
                sample.features[j] =
                    (sample.features[j] - globalMeans[j])
                    / standardDeviation;
            }
            else
            {
                sample.features[j] = 0.0;
            }
        }
    }

    cout << "\nFeature standardization completed." << endl;
}

map<int, double> CalculateClassPriors(const Dataset& dataset)
{
    map<int, int> classCounts;
    map<int, double> classPriors;

    int N = static_cast<int>(dataset.samples.size());

    if (N == 0)
    {
        return classPriors;
    }

    // Count observations belonging to each class
    for (const Sample& sample : dataset.samples)
    {
        classCounts[sample.label]++;
    }

    // Calculate class prior probabilities
    for (const auto& entry : classCounts)
    {
        int label = entry.first;
        int count = entry.second;

        classPriors[label] =
            static_cast<double>(count) / N;
    }

    cout << "\nClass Distributions:" << endl;
    cout << "Class\tCount\tPrior Probability" << endl;

    for (const auto& entry : classCounts)
    {
        int label = entry.first;
        int count = entry.second;

        cout << label << "\t"
             << count << "\t"
             << fixed << setprecision(6)
             << classPriors[label] << endl;
    }

    return classPriors;
}

map<int, FeatureVector> CalculateClassMeans(
    const Dataset& dataset)
{
    map<int, FeatureVector> classMeans;
    map<int, int> classCounts;

    int numFeatures = dataset.numFeatures;

    // Accumulate feature sums for each class
    for (const Sample& sample : dataset.samples)
    {
        int label = sample.label;

        if (classMeans.find(label) == classMeans.end())
        {
            classMeans[label] =
                FeatureVector(numFeatures, 0.0);
        }

        classCounts[label]++;

        for (int j = 0; j < numFeatures; j++)
        {
            classMeans[label][j] += sample.features[j];
        }
    }

    // Divide feature sums by class counts
    for (auto& entry : classMeans)
    {
        int label = entry.first;
        int count = classCounts[label];

        for (int j = 0; j < numFeatures; j++)
        {
            entry.second[j] /= count;
        }
    }

    return classMeans;
}

void RegularizeCovariance(
    CovarianceMatrix& covariance,
    double lambda)
{
    int numFeatures =
        static_cast<int>(covariance.size());

    for (int i = 0; i < numFeatures; i++)
    {
        covariance[i][i] += lambda;
    }
}

double CalculateTrace(const CovarianceMatrix& covariance)
{
    double trace = 0.0;

    int numFeatures =
        static_cast<int>(covariance.size());

    for (int i = 0; i < numFeatures; i++)
    {
        trace += covariance[i][i];
    }

    return trace;
}

double CalculateLambda(
    const CovarianceMatrix& covariance,
    int classCount,
    double alpha)
{
    int numFeatures =
        static_cast<int>(covariance.size());

    if (numFeatures == 0 || classCount <= 1)
    {
        return 0.0;
    }

    int rankUpperBound =
        min(numFeatures, classCount - 1);

    double trace = CalculateTrace(covariance);

    double lambda =
        alpha * trace / rankUpperBound;

    return lambda;
}

map<int, CovarianceMatrix> CalculateClassCovariances(
    const Dataset& dataset,
    const map<int, FeatureVector>& classMeans)
{
    map<int, CovarianceMatrix> classCovariances;
    map<int, int> classCounts;

    int numFeatures = dataset.numFeatures;

    // Initialize covariance matrices
    for (const auto& entry : classMeans)
    {
        int label = entry.first;

        classCovariances[label] = CovarianceMatrix(
            numFeatures,
            FeatureVector(numFeatures, 0.0)
        );
    }

    // Accumulate covariance sums
    for (const Sample& sample : dataset.samples)
    {
        int label = sample.label;

        classCounts[label]++;

        const FeatureVector& mean = classMeans.at(label);
        CovarianceMatrix& covariance =
            classCovariances.at(label);

        for (int i = 0; i < numFeatures; i++)
        {
            double differenceI =
                sample.features[i] - mean[i];

            for (int j = 0; j < numFeatures; j++)
            {
                double differenceJ =
                    sample.features[j] - mean[j];

                covariance[i][j] +=
                    differenceI * differenceJ;
            }
        }
    }

    // Divide by the class observation counts
    for (auto& entry : classCovariances)
    {
        int label = entry.first;
        int count = classCounts[label];

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                entry.second[i][j] /= count;
            }
        }
    }

    return classCovariances;
}

bool CholeskyDecomposition(
    const CovarianceMatrix& covariance,
    CovarianceMatrix& lower)
{
    int numFeatures =
        static_cast<int>(covariance.size());

    lower = CovarianceMatrix(
        numFeatures,
        FeatureVector(numFeatures, 0.0)
    );

    for (int i = 0; i < numFeatures; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = covariance[i][j];

            for (int k = 0; k < j; k++)
            {
                sum -= lower[i][k] * lower[j][k];
            }

            if (i == j)
            {
                if (sum <= 0.0 || !isfinite(sum))
                {
                    return false;
                }

                lower[i][j] = sqrt(sum);
            }
            else
            {
                lower[i][j] =
                    sum / lower[j][j];
            }
        }
    }

    return true;
}

double CalculateLogDeterminant(
    const CovarianceMatrix& lower)
{
    double logDeterminant = 0.0;

    int numFeatures =
        static_cast<int>(lower.size());

    for (int i = 0; i < numFeatures; i++)
    {
        logDeterminant += log(lower[i][i]);
    }

    return 2.0 * logDeterminant;
}

vector<GaussianClass> TrainGaussianModels(
    const Dataset& dataset,
    double alpha)
{
    vector<GaussianClass> models;

    if (dataset.samples.empty())
    {
        return models;
    }

    // Estimate class priors
    map<int, double> classPriors =
        CalculateClassPriors(dataset);

    // Estimate class mean vectors
    map<int, FeatureVector> classMeans =
        CalculateClassMeans(dataset);

    // Estimate class covariance matrices
    map<int, CovarianceMatrix> classCovariances =
        CalculateClassCovariances(dataset, classMeans);

    // Count observations in each class
    map<int, int> classCounts;

    for (const Sample& sample : dataset.samples)
    {
        classCounts[sample.label]++;
    }

    // Construct one Gaussian model per class
    for (const auto& entry : classPriors)
    {
        int label = entry.first;

        GaussianClass model;

        model.label = label;
        model.count = classCounts.at(label);
        model.prior = entry.second;
        model.mean = classMeans.at(label);
        model.covariance = classCovariances.at(label);

        // Calculate regularization parameter
        model.lambda = CalculateLambda(
            model.covariance,
            model.count,
            alpha
        );

        // Apply covariance regularization
        RegularizeCovariance(
            model.covariance,
            model.lambda
        );

        // Calculate and store Cholesky factor
        bool success = CholeskyDecomposition(
            model.covariance,
            model.lower
        );

        if (!success)
        {
            cerr << "Cholesky decomposition failed for class "
                 << model.label << endl;
            
            return {};
        }

        // Calculate and store log determinant
        model.logDeterminant = CalculateLogDeterminant(model.lower);

        models.push_back(model);
    }

    return models;
}

double CalculateMahalanobisDistance(
    const FeatureVector& features,
    const FeatureVector& mean,
    const CovarianceMatrix& lower)
{
    int numFeatures = static_cast<int>(features.size());

    FeatureVector y(numFeatures, 0.0);

    // Solve L * y = x - mean using forward sub
    for (int i = 0; i < numFeatures; i++)
    {
        double sum = features[i] - mean[i];

        for (int j = 0; j < i; j++)
        {
            sum -= lower[i][j] * y[j];
        }

        y[i] = sum / lower[i][i];
    }

    // Calculate squared Mahalanobis distance
    double distanceSquared = 0.0;

    for (int i = 0; i < numFeatures; i++)
    {
        distanceSquared += y[i] * y[i];
    }

    return distanceSquared;
}

double CalculateDiscriminantScore(
    const FeatureVector& features,
    const GaussianClass& model)
{
    double distanceSquared = CalculateMahalanobisDistance(
        features,
        model.mean,
        model.lower
    );

    double score = log(model.prior) - 0.5 * model.logDeterminant
     - 0.5 * distanceSquared;
    
    return score;
}

int PredictClass(
    const FeatureVector& features,
    const vector<GaussianClass>& models)
{
    if (models.empty())
    {
        return -1;
    }

    int predictedClass = models[0].label;

    double highestScore = CalculateDiscriminantScore(
        features,
        models[0]
    );

    // Compare discriminant score across all classes
    for (int k = 1; k < static_cast<int>(models.size()); k++)
    {
        double score = CalculateDiscriminantScore(
            features,
            models[k]
        );

        if (score > highestScore)
        {
            highestScore = score;
            predictedClass = models[k].label;
        }
    }

    return predictedClass;
}

void EvaluateClassifier(
    const Dataset& dataset,
    const vector<GaussianClass>& models)
{
    int numClasses = static_cast<int>(models.size());

    if (numClasses == 0 || dataset.samples.empty())
    {
        cerr << "Cannot evaluate an empty dataset or model."
             << endl;
        return;
    }

    // Map class labels to confusion matrix indices
    map<int, int> classIndices;

    for (int i = 0; i < numClasses; i++)
    {
        classIndices[models[i].label] = i;
    }

    // Initialize confusion matrix
    vector<vector<int>> confusionMatrix(
        numClasses,
        vector<int>(numClasses, 0)
    );

    int incorrect = 0;

    // Classify every observation
    for (const Sample& sample : dataset.samples)
    {
        int predictedClass = PredictClass(
            sample.features,
            models
        );

        int actualIndex = classIndices.at(sample.label);
        int predictedIndex = classIndices.at(predictedClass);

        confusionMatrix[actualIndex][predictedIndex]++;

        if (predictedClass != sample.label)
        {
            incorrect++;
        }
    }

    int total = static_cast<int>(dataset.samples.size());

    double trainingError = static_cast<double>(incorrect) / total;

    // Print confusion matrix
    cout << "\nConfusion Matrix:" << endl;
    cout << "Rows = Actual, Columns = Predicted" << endl;

    cout << setw(10) << "Actual";

    for (const GaussianClass& model : models)
    {
        cout << setw(10) << model.label;
    }

    cout << endl;

    for (int i = 0; i < numClasses; i++)
    {
        cout << setw(10) << models[i].label;

        for (int j = 0; j < numClasses; j++)
        {
            cout << setw(10) << confusionMatrix[i][j];
        }

        cout << endl;
    }

    cout << "\nTotal Observations: " << total << endl;
    cout << "Correct Classification: "
         << total - incorrect << endl;
    cout << "Incorrect Classification: "
         << incorrect << endl;
    
    cout << "Empirical Training Error: "
         << fixed << setprecision(6)
         << trainingError << endl;
    
    cout << "Training Accuracy: "
         << fixed << setprecision(2)
         << (1.0 - trainingError) * 100.0
         << "%" << endl;
}

void ExportVisualizationData(
    const Dataset& dataset,
    const string& filename)
{
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "Could not create file: "
             << filename << endl;
        return;
    }

    // Write column headers
    for (int j = 0; j < dataset.numFeatures; j++)
    {
        file << "Feature" << j + 1 << ",";
    }

    file << "Class" << endl;

    // Export all standardized observations
    for (const Sample& sample : dataset.samples)
    {
        for (int j = 0; j < dataset.numFeatures; j++)
        {
            file << sample.features[j] << ",";
        }

        file << sample.label << endl;
    }

    file.close();

    cout << "Visualization data exported to "
         << filename << endl;
}

void Generate2DPlot(
    const Dataset& dataset,
    const string& dataFilename,
    const string& scriptFilename,
    const string& imageFilename,
    const string& title,
    int featureX,
    int featureY)
{
    // Validate feature indices
    if (featureX < 0 || featureY < 0 ||
        featureX >= dataset.numFeatures ||
        featureY >= dataset.numFeatures)
    {
        cerr << "Invalid feature indices." << endl;
        return;
    }

    // Determine class label column
    int classColumn = dataset.numFeatures + 1;

    // Generate Gnuplot script
    ofstream scriptFile(scriptFilename);

    if (!scriptFile.is_open())
    {
        cerr << "Error creating Gnuplot script." << endl;
        return;
    }

    // Configure plot
    scriptFile << "set terminal pngcairo size 1000,750\n";
    scriptFile << "set output '" << imageFilename << "'\n";
    scriptFile << "set title '" << title << "'\n";
    scriptFile << "set xlabel 'Standardized Feature "
               << featureX + 1 << "'\n";
    scriptFile << "set ylabel 'Standardized Feature "
               << featureY + 1 << "'\n";
    scriptFile << "set grid\n";
    scriptFile << "set key outside right\n";
    scriptFile << "set datafile separator ','\n";
    scriptFile << "set datafile columnheaders\n";

    // Collect all class labels
    map<int, bool> classes;

    for (const Sample& sample : dataset.samples)
    {
        classes[sample.label] = true;
    }

    // Plot each class separately
    scriptFile << "plot ";

    bool first = true;

    for (const auto& entry : classes)
    {
        if (!first)
        {
            scriptFile << ", \\\n";
        }

        scriptFile << "'" << dataFilename
                   << "' using " << featureX + 1
                   << ":($" << classColumn << "=="
                   << entry.first
                   << "?$" << featureY + 1 << ":1/0) "
                   << "with points pointtype 7 "
                   << "pointsize 0.4 "
                   << "title 'Class "
                   << entry.first << "'";
        
        first = false;
    }

    scriptFile << "\n";
    scriptFile.close();

    cout << "Generated Gnuplot script: "
         << scriptFilename << endl;
}

void Generate3DPlot(
    const Dataset& dataset,
    const string& dataFilename,
    const string& scriptFilename,
    const string& imageFilename,
    const string& title,
    int featureX,
    int featureY,
    int featureZ)
{
    // Validate feature indices
    if (featureX < 0 || featureY < 0 || featureZ < 0 ||
        featureX >= dataset.numFeatures ||
        featureY >= dataset.numFeatures ||
        featureZ >= dataset.numFeatures)
    {
        cerr << "Invalid feature indices." << endl;
        return;
    }

    // Determine class label column
    int classColumn = dataset.numFeatures + 1;

    // Generate Gnuplot script
    ofstream scriptFile(scriptFilename);

    if (!scriptFile.is_open())
    {
        cerr << "Error creating Gnuplot script." << endl;
        return;
    }

    // Configure plot
    scriptFile << "set terminal pngcairo size 1000,750\n";
    scriptFile << "set output '" << imageFilename << "'\n";
    scriptFile << "set title '" << title << "'\n";
    scriptFile << "set xlabel 'Standardized Feature "
               << featureX + 1 << "'\n";
    scriptFile << "set ylabel 'Standardized Feature "
               << featureY + 1 << "'\n";
    scriptFile << "set zlabel 'Standardized Feature "
               << featureZ + 1 << "'\n";
    scriptFile << "set grid\n";
    scriptFile << "set key outside right\n";
    scriptFile << "set datafile separator ','\n";
    scriptFile << "set datafile columnheaders\n";
    scriptFile << "set view 60,30\n";
    scriptFile << "set lmargin 12\n";
    scriptFile << "set zlabel offset 2,0\n";

    // Collect all class labels
    map<int, bool> classes;

    for (const Sample& sample : dataset.samples)
    {
        classes[sample.label] = true;
    }

    // Plot each class separately
    scriptFile << "splot ";

    bool first = true;

    for (const auto& entry : classes)
    {
        if (!first)
        {
            scriptFile << ", \\\n";
        }

        scriptFile << "'" << dataFilename
                   << "' using " << featureX + 1
                   << ":" << featureY + 1
                   << ":(column(" << classColumn << ")=="
                   << entry.first
                   << "?column(" << featureZ + 1 
                   << "):1/0) "
                   << "with points pointtype 7 "
                   << "pointsize 0.4 "
                   << "title 'Class "
                   << entry.first << "'";
        
        first = false;
    }

    scriptFile << "\n";
    scriptFile.close();

    cout << "Generated Gnuplot script: "
         << scriptFilename << endl;
}


int main()
{
    const double ALPHA = 0.01;

    // Load Wine Quality dataset
    Dataset wine = LoadWineDataset(
        "wine+quality/winequality-white.csv"
    );

    if (wine.samples.empty())
    {
        cerr << "Error: Wine Quality dataset is empty."
             << endl;
        return 1;
    }

    cout << "\nQuestion 3: Wine Quality Classification"
         << endl;

    cout << "Total observations: "
         << wine.samples.size() << endl;

    cout << "Number of features: "
         << wine.numFeatures << endl;

    // Standardize features and train Gaussian models
    StandardizeFeatures(wine);

    vector<GaussianClass> wineModels =
        TrainGaussianModels(wine, ALPHA);

    if (wineModels.empty())
    {
        cerr << "Error: Wine model training failed."
             << endl;
        return 1;
    }

    // Display class priors and regularization parameters
    cout << "\nWine Quality Gaussian Models:" << endl;

    for (const GaussianClass& model : wineModels)
    {
        cout << "Class " << model.label
             << ": Count = " << model.count
             << ", Prior = " << fixed << setprecision(6)
             << model.prior
             << ", Lambda = " << model.lambda
             << endl;
    }

    // Evaluate Wine Quality classifier
    cout << "\nWine Quality Classification Results:"
         << endl;

    EvaluateClassifier(wine, wineModels);

    // Load Human Activity Recognition dataset
    Dataset har = LoadHARDataset(
        "human+activity+recognition+using+smartphones/"
        "UCI HAR Dataset"
    );

    if (har.samples.empty())
    {
        cerr << "Error: HAR dataset is empty." << endl;
        return 1;
    }

    cout << "\nQuestion 3: Human Activity Recognition"
         << endl;

    cout << "Total observations: "
         << har.samples.size() << endl;

    cout << "Number of features: "
         << har.numFeatures << endl;

    // Standardize features and train Gaussian models
    StandardizeFeatures(har);

    vector<GaussianClass> harModels =
        TrainGaussianModels(har, ALPHA);

    if (harModels.empty())
    {
        cerr << "Error: HAR model training failed."
             << endl;
        return 1;
    }

    // Display class priors and regularization parameters
    cout << "\nHAR Gaussian Models:" << endl;

    for (const GaussianClass& model : harModels)
    {
        cout << "Class " << model.label
             << ": Count = " << model.count
             << ", Prior = " << fixed << setprecision(6)
             << model.prior
             << ", Lambda = " << model.lambda
             << endl;
    }

    // Evaluate HAR classifier
    cout << "\nHuman Activity Recognition Classification Results:"
         << endl;

    EvaluateClassifier(har, harModels);

    // Export standardized visualization data
    ExportVisualizationData(
        wine,
        "wine_visualization.csv"
    );

    ExportVisualizationData(
        har,
        "har_visualization.csv"
    );

    // Generate Wine Quality 2D plot
    Generate2DPlot(
        wine,
        "wine_visualization.csv",
        "Question3_Wine_2D.gp",
        "Question3_Wine_2D.png",
        "Wine Quality - 2D Feature Distribution",
        0,
        1
    );

    // Generate HAR 2D plot
    Generate2DPlot(
        har,
        "har_visualization.csv",
        "Question3_HAR_2D.gp",
        "Question3_HAR_2D.png",
        "HAR - 2D Feature Distribution",
        0,
        1
    );

    // Generate Wine Quality 3D plot
    Generate3DPlot(
        wine,
        "wine_visualization.csv",
        "Question3_Wine_3D.gp",
        "Question3_Wine_3D.png",
        "Wine Quality - 3D Feature Distribution",
        0,
        1,
        2
    );

    // Generate HAR 3D plot
    Generate3DPlot(
        har,
        "har_visualization.csv",
        "Question3_HAR_3D.gp",
        "Question3_HAR_3D.png",
        "HAR - 3D Feature Distribution",
        0,
        1,
        2
    );

    return 0;
}


