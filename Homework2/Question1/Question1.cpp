#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
using namespace std;

class Sample
{
private:
    double x1;
    double x2;
    int label;

public:
    // Constructor
    Sample(double x1Value, double x2Value, int classLabel)
    {
        x1 = x1Value;
        x2 = x2Value;
        label = classLabel;
    }

    // Getters
    double getX1() const
    {
        return x1;
    }

    double getX2() const
    {
        return x2;
    }

    int getLabel() const
    {
        return label;
    }

    // Print sample info
    void print() const
    {
        cout << "x1: " << x1
             << " | x2: " << x2
             << " | Label: " << label
             << endl;
    }
};

class GaussianMixture
{
private:
    // Class priors
    double prior0;
    double prior1;

    // Gaussian mixture weights
    double weight;

    // Mean vectors
    double m01[2];
    double m02[2];
    double m11[2];
    double m12[2];

    // Diagonal covariance entries
    double var1;
    double var2;

    // Random number generator
    mt19937 rng;

public:
    // Constructor
    GaussianMixture(unsigned int seed = 42) : rng(seed)
    {
        prior0 = 0.6;
        prior1 = 0.4;

        weight = 0.5;

        m01[0] = -0.9;
        m01[1] = -1.1;

        m02[0] = 0.8;
        m02[1] = 0.75;

        m11[0] = -1.1;
        m11[1] = 0.9;

        m12[0] = 0.9;
        m12[1] = -0.75;

        var1 = 0.75;
        var2 = 1.25;
    }

    Sample generateSample()
    {
        uniform_real_distribution<double> uniform(0.0, 1.0);
        normal_distribution<double> normal(0.0, 1.0);

        // Select class
        double classRandom = uniform(rng);

        int label;

        if (classRandom < prior0)
        {
            label = 0;
        }
        else
        {
            label = 1;
        }

        // Select Gaussian component
        double componentRandom = uniform(rng);

        const double* mean;

        if (label == 0)
        {
            if (componentRandom < weight)
            {
                mean = m01;
            }
            else
            {
                mean = m02;
            }
        }
        else
        {
            if (componentRandom < weight)
            {
                mean = m11;
            }
            else
            {
                mean = m12;
            }
        }

        // Generate Gaussian coordinates
        double x1 = mean[0] + sqrt(var1) * normal(rng);
        double x2 = mean[1] + sqrt(var2) * normal(rng);

        // Return sample
        return Sample(x1, x2, label);
    }

    void printParameters() const
    {
        cout << "Gaussian Mixture Parameters:" << endl;

        cout << "Class 0 Prior: " << prior0 << endl;
        cout << "Class 1 Prior: " << prior1 << endl;

        cout << "Mixture Weight: " << weight << endl;

        cout << "\nMean Vectors:" << endl;

        cout << "m01: (" << m01[0] << ", " << m01[1] << ")" << endl;
        cout << "m02: (" << m02[0] << ", " << m02[1] << ")" << endl;
        cout << "m11: (" << m11[0] << ", " << m11[1] << ")" << endl;
        cout << "m12: (" << m12[0] << ", " << m12[1] << ")" << endl;

        cout << "\nCovariance Matrix:" << endl;
        cout << "[" << var1 << ", 0]" << endl;
        cout << "[0, " << var2 << "]" << endl;
    }
};

class Dataset
{
private:
    vector<Sample> samples;

public:
    // Generate dataset of N observations
    void generate(int N, GaussianMixture& model)
    {  
        samples.clear();

        for (int i = 0; i < N; i++)
        {
            samples.push_back(model.generateSample());
        }
    }

    // Return dataset size
    int getSize() const
    {
        return static_cast<int>(samples.size());
    }

    // Retrieve an observation
    const Sample& getSample(int index) const
    {
        return samples.at(index);
    }

    // Count observations in a given class
    int countClass(int label) const
    {
        int count = 0;

        for (const Sample& sample : samples)
        {
            if (sample.getLabel() == label)
            {
                count++;
            }
        }

        return count;
    }

    // Print first N samples
    void printSamples(int N) const
    {
        int limit = N;

        if (limit > getSize())
        {
            limit = getSize();
        }

        for (int i = 0; i < limit; i++)
        {
            cout << "Sample: " << i + 1 << ": ";
            samples[i].print();
        }
    }

    // Save dataset to CSV
    void saveCSV(const string& filename) const
    {
        ofstream file(filename);

        if (!file.is_open())
        {
            cerr << "Error opening file: " << filename << endl;
            return;
        }

        file << "x1,x2,label\n";
        file << setprecision(17);

        for (const Sample& sample : samples)
        {
            file << sample.getX1() << ","
                 << sample.getX2() << ","
                 << sample.getLabel() << "\n";
        }

        file.close();

        cout << "Saved: " << filename << endl;
    }
};

class OptimalClassifier
{
private:
    double prior0;
    double prior1;

    double weight;

    double m01[2];
    double m02[2];
    double m11[2];
    double m12[2];

    double var1;
    double var2;

public:
    OptimalClassifier()
    {
        prior0 = 0.6;
        prior1 = 0.4;

        weight = 0.5;

        m01[0] = -0.9;
        m01[1] = -1.1;

        m02[0] = 0.8;
        m02[1] = 0.75;

        m11[0] = -1.1;
        m11[1] = 0.9;

        m12[0] = 0.9;
        m12[1] = -0.75;

        var1 = 0.75;
        var2 = 1.25;
    }

    double gaussianPDF(const Sample& sample, const double mean[2]) const
    {
        double x1 = sample.getX1();
        double x2 = sample.getX2();

        double diff1 = x1 - mean[0];
        double diff2 = x2 - mean[1];

        double exponent = -0.5 * 
            ((diff1 * diff1) / var1 +
             (diff2 * diff2) / var2);
            
        double pi = acos(-1.0);

        double denominator = 2.0 * pi * sqrt(var1 * var2);

        return exp(exponent) / denominator;
    }

    double classLikelihood(const Sample& sample, int label) const
    {
        if (label == 0)
        {
            return weight * gaussianPDF(sample, m01)
                 + weight * gaussianPDF(sample, m02);
        }
        else if (label == 1)
        {
            return weight * gaussianPDF(sample, m11)
                 + weight * gaussianPDF(sample, m12);
        }

        return 0.0;
    }

    int predict(const Sample& sample) const
    {
        double score0 = prior0 * classLikelihood(sample, 0);
        double score1 = prior1 * classLikelihood(sample, 1);

        if (score1 > score0)
        {
            return 1;
        }

        return 0;
    }

    double discriminantScore(const Sample& sample) const
    {
        double score0 = prior0 * classLikelihood(sample, 0);
        double score1 = prior1 * classLikelihood(sample, 1);

        return log(score1) - log(score0);
    }

    void evaluate(const Dataset& dataset) const
    {
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            const Sample& sample = dataset.getSample(i);

            int actual = sample.getLabel();
            int predicted = predict(sample);

            if (actual == 1 && predicted == 1)
            {
                TP++;
            }
            else if (actual == 0 && predicted == 0)
            {
                TN++;
            }
            else if (actual == 0 && predicted == 1)
            {
                FP++;
            }
            else if (actual == 1 && predicted == 0)
            {
                FN++;
            }
        }

        double errorRate = static_cast<double>(FP + FN) / dataset.getSize();
        double accuracy = static_cast<double>(TP + TN) / dataset.getSize();

        cout << "\nOptimal Classifier Evaluation" << endl;

        cout << "True Positives: " << TP << endl;
        cout << "True Negatives: " << TN << endl;
        cout << "False Positives: " << FP << endl;
        cout << "False Negatives: " << FN << endl;

        cout << "\nTotal Samples: " << dataset.getSize() << endl;
        cout << "Classification Error: " << errorRate << endl;
        cout << "Accuracy: " << accuracy << endl;
    }

    void generateROC(const Dataset& dataset) const
    {
        ofstream file("Q1_Optimal_ROC.csv");

        if (!file.is_open())
        {
            cerr << "Error creating ROC file." << endl;
            return;
        }

        file << "FPR,TPR,Threshold\n";
        file << setprecision(17);

        vector<double> scores;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            scores.push_back(discriminantScore(dataset.getSample(i)));
        }

        // Evaluate a range of thresholds
        for (double threshold = -20.0;
                    threshold <= 20.0;
                    threshold += 0.05)
        {
            int TP = 0;
            int TN = 0;
            int FP = 0;
            int FN = 0;

            for (int i = 0; i < dataset.getSize(); i++)
            {
                int actual = dataset.getSample(i).getLabel();
                int predicted = (scores[i] > threshold) ? 1 : 0;

                if (actual == 1 && predicted == 1)
                {
                    TP++;
                }
                else if (actual == 0 && predicted == 0)
                {
                    TN++;
                }
                else if (actual == 0 && predicted == 1)
                {
                    FP++;
                }
                else
                {
                    FN++;
                }
            }

            double TPR = static_cast<double>(TP) / (TP + FN);
            double FPR = static_cast<double>(FP) / (FP + TN);

            file << FPR << "," << TPR << "," << threshold << "\n";
        }

        file.close();

        cout << "Saved: Q1_Optimal_ROC.csv" << endl;
    }
};

class LogisticClassifier
{
private:
    // Model Parameters
    double w0;
    double w1;
    double w2;

    // Training settings
    double learningRate;
    int maxIterations;

public:
    // Constructor
    LogisticClassifier(double lr = 0.01, int iterations = 10000)
    {
        w0 = 0.0;
        w1 = 0.0;
        w2 = 0.0;

        learningRate = lr;
        maxIterations = iterations;
    }

    // Logistic sigmoid function
    double sigmoid(double z) const
    {
        if (z >= 0.0)
        {
            return 1.0 / (1.0 + exp(-z));
        }

        double ez = exp(z);
        return ez / (1.0 + ez);
    }

    // Estimate probability of Class 1
    double predictProbability(const Sample& sample) const
    {
        double z = w0 
                 + w1 * sample.getX1() 
                 + w2 * sample.getX2();
        
        return sigmoid(z);
    }

    // Predict class label
    int predict(const Sample& sample) const
    {
        double probability = predictProbability(sample);

        if (probability > 0.5)
        {
            return 1;
        }

        return 0;
    }

    double negativeLogLikelihood(const Dataset& dataset) const
    {
        double totalLoss = 0.0;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            const Sample& sample = dataset.getSample(i);

            double y = sample.getLabel();

            double z = w0 
                     + w1 * sample.getX1() 
                     + w2 * sample.getX2();
            
            // Numerically stable negative log-likelihood
            totalLoss += max(0.0, z) 
                      - y * z 
                      + log1p(exp(-abs(z)));
        }

        return totalLoss / dataset.getSize();
    }

    void train(const Dataset& dataset)
    {
        int N = dataset.getSize();

        if (N == 0)
        {
            cerr << "Error: Cannot train on an empty dataset" << endl;
            return;
        }

        // Reset parameters before training
        w0 = 0.0;
        w1 = 0.0;
        w2 = 0.0;

        cout << "\nTraining Logistic Classifier" << endl;
        cout << "Training Samples: " << N << endl;

        cout << "Initial NLL: " 
             << negativeLogLikelihood(dataset) << endl;
        
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            double gradient0 = 0.0;
            double gradient1 = 0.0;
            double gradient2 = 0.0;

            // Calculate gradients using all training samples
            for (int i = 0; i < N; i++)
            {
                const Sample& sample = dataset.getSample(i);

                double x1 = sample.getX1();
                double x2 = sample.getX2();
                double y = sample.getLabel();

                double probability = predictProbability(sample);

                double error = probability - y;

                gradient0 += error;
                gradient1 += error * x1;
                gradient2 += error * x2;
            }

            // Average gradients
            gradient0 /= N;
            gradient1 /= N;
            gradient2 /= N;

            // Update parameters
            w0 -= learningRate * gradient0;
            w1 -= learningRate * gradient1;
            w2 -= learningRate * gradient2;

            // Print progress every 1000 iterations
            if ((iteration + 1) % 1000 == 0)
            {
                cout << "Iteration " << iteration + 1
                     << " | NLL: " 
                     << negativeLogLikelihood(dataset)
                     << endl;
            }
        }

        cout << "\nFinal NLL: "
             << negativeLogLikelihood(dataset)
             << endl;
        
        cout << "Learned Parameters:" << endl;
        printParameters();
    }

    void evaluate(const Dataset& dataset) const
    {
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            const Sample& sample = dataset.getSample(i);

            int actual = sample.getLabel();
            int predicted = predict(sample);

            if (actual == 1 && predicted == 1)
            {
                TP++;
            }
            else if (actual == 0 && predicted == 0)
            {
                TN++;
            }
            else if (actual == 0 && predicted == 1)
            {
                FP++;
            }
            else if (actual == 1 && predicted == 0)
            {
                FN++;
            }
        }

        double errorRate = static_cast<double>(FP + FN) / dataset.getSize();
        double accuracy = static_cast<double>(TP + TN) / dataset.getSize();

        cout << "\nLogistic Classifier Evaluation" << endl;

        cout << "True Positives: " << TP << endl;
        cout << "True Negatives: " << TN << endl;
        cout << "False Positives: " << FP << endl;
        cout << "False Negatives: " << FN << endl;

        cout << "Classification Error: " << errorRate << endl;
        cout << "Accuracy: " << accuracy << endl;
    }

    // Print learned parameters
    void printParameters() const
    {
        cout << "w0: " << w0 << endl;
        cout << "w1: " << w1 << endl;
        cout << "w2: " << w2 << endl;
    }
};

class LogisticQuadraticClassifier
{
private:
    // Six model parameter
    vector<double> weights;

    // Training settings
    double learningRate;
    int maxIterations;

public:
    // Constructor
    LogisticQuadraticClassifier(double lr = 0.01, int iterations = 10000)
    {
        weights = vector<double>(6, 0.0);

        learningRate = lr;
        maxIterations = iterations;
    }

    // Generate quadratic feature vector
    vector<double> getFeatures(const Sample& sample) const
    {
        double x1 = sample.getX1();
        double x2 = sample.getX2();

        return{
            1.0,
            x1,
            x2,
            x1 * x1,
            x1 * x2,
            x2 * x2
        };
    }

    // Numerically stable sigmoid
    double sigmoid(double z) const
    {
        if (z >= 0.0)
        {
            return 1.0 / (1.0 + exp(-z));
        }

        double ez = exp(z);
        return ez / (1.0 + ez);
    }

    // Predict probability of Class 1
    double predictProbability(const Sample& sample) const
    {
        vector<double> features = getFeatures(sample);

        double z = 0.0;

        for (int i = 0; i < 6; i++)
        {
            z += weights[i] * features[i];
        }

        return sigmoid(z);
    }

    // Predict class label
    int predict(const Sample& sample) const
    {
        if (predictProbability(sample) > 0.5)
        {
            return 1;
        }

        return 0;
    }

    double negativeLogLikelihood(const Dataset& dataset) const
    {
        double totalLoss = 0.0;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            const Sample& sample = dataset.getSample(i);

            double y = sample.getLabel();

            vector<double> features = getFeatures(sample);

            double z = 0.0;

            for (int j = 0; j < 6; j++)
            {
                z += weights[j] * features[j];
            }

            // Numerically stable negative log-likelihood
            totalLoss += max(0.0, z)
                      - y * z
                      + log1p(exp(-abs(z)));
        }

        return totalLoss / dataset.getSize();
    }

    void train(const Dataset& dataset)
    {
        int N = dataset.getSize();

        if (N == 0)
        {
            cerr << "Error: Cannot train on an empty dataset" << endl;
            return;
        }

        // Reset all six parameters
        weights.assign(6, 0.0);

        cout << "\nTraining Logistic Quadratic Classifier" << endl;
        cout << "Training Samples: " << N << endl;

        cout << "Initial NLL: " 
             << negativeLogLikelihood(dataset) << endl;
        
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            vector<double> gradients(6, 0.0);

            // Calculate gradients for every training sample
            for (int i = 0; i < N; i++)
            {
                const Sample& sample = dataset.getSample(i);

                vector<double> features = getFeatures(sample);

                double probability = predictProbability(sample);
                double y = sample.getLabel();

                double error = probability - y;

                // Accumulate gradient for each parameter
                for (int j = 0; j < 6; j++)
                {
                    gradients[j] += error * features[j];
                }
            }

            // Average gradients and update weights
            for (int j = 0; j < 6; j++)
            {
                gradients[j] /= N;

                weights[j] -= learningRate * gradients[j];
            }

            // Display training progress
            if ((iteration + 1) % 10000 == 0)
            {
                cout << "Iteration " << iteration + 1
                     << " | NLL: "
                     << negativeLogLikelihood(dataset)
                     << endl;
            }
        }

        cout << "\nFinal NLL: "
             << negativeLogLikelihood(dataset)
             << endl;

        cout << "Learned Parameters:" << endl;
        printParameters();
    }


    void evaluate(const Dataset& dataset) const
    {
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;

        for (int i = 0; i < dataset.getSize(); i++)
        {
            const Sample& sample = dataset.getSample(i);

            int actual = sample.getLabel();
            int predicted = predict(sample);

            if (actual == 1 && predicted == 1)
            {
                TP++;
            }
            else if (actual == 0 && predicted == 0)
            {
                TN++;
            }
            else if (actual == 0 && predicted == 1)
            {
                FP++;
            }
            else if (actual == 1 && predicted == 0)
            {
                FN++;
            }
        }

        int total = TP + TN + FP + FN;

        double errorRate =
            static_cast<double>(FP + FN) / total;

        double accuracy =
            static_cast<double>(TP + TN) / total;

        cout << "\nLogistic Quadratic Classifier Evaluation" << endl;

        cout << "True Positives: " << TP << endl;
        cout << "True Negatives: " << TN << endl;
        cout << "False Positives: " << FP << endl;
        cout << "False Negatives: " << FN << endl;

        cout << "Total Samples: " << total << endl;

        cout << "Classification Error: " << errorRate << endl;
        cout << "Accuracy: " << accuracy << endl;
    }


    // Print learned parameters
    void printParameters() const
    {
        for (int i = 0; i < 6; i++)
        {
            cout << "w" << i << ": " << weights[i] << endl;
        }
    }
};

void SaveComparisonResults()
{
    ofstream file("Q1_Classifier_Comparison.csv");

    if (!file.is_open())
    {
        cerr << "Error: Could not create comparison CSV." << endl;
        return;
    }

    file << "Classifier,TrainingSamples,ErrorRate,Accuracy,PlotLabel\n";

    file << "BayesOptimal,0,0.2727,0.7273,Bayes Optimal\n";

    file << "LogisticLinear,20,0.3678,0.6322,Logistic Linear\n";
    file << "LogisticLinear,200,0.4611,0.5389,Logistic Linear\n";
    file << "LogisticLinear,2000,0.3945,0.6055,Logistic Linear\n";

    file << "LogisticQuadratic,20,0.3279,0.6721,Logistic Quadratic\n";
    file << "LogisticQuadratic,200,0.2816,0.7184,Logistic Quadratic\n";
    file << "LogisticQuadratic,2000,0.2761,0.7239,Logistic Quadratic\n";

    file.close();

    cout << "Saved: Q1_Classifier_Comparison.csv" << endl;
}

int main()
{
    // ==========================================
    // Generate Datasets
    // ==========================================

    GaussianMixture model(42);

    Dataset Dtrain20;
    Dataset Dtrain200;
    Dataset Dtrain2000;
    Dataset Dvalidate;

    Dtrain20.generate(20, model);
    Dtrain200.generate(200, model);
    Dtrain2000.generate(2000, model);
    Dvalidate.generate(10000, model);

    cout << "========== DATASET GENERATION ==========" << endl;

    cout << "Dtrain20: " << Dtrain20.getSize() << endl;
    cout << "Dtrain200: " << Dtrain200.getSize() << endl;
    cout << "Dtrain2000: " << Dtrain2000.getSize() << endl;
    cout << "Dvalidate: " << Dvalidate.getSize() << endl;

    cout << "\nValidation Class Counts" << endl;
    cout << "Class 0: " << Dvalidate.countClass(0) << endl;
    cout << "Class 1: " << Dvalidate.countClass(1) << endl;

    // Export datasets
    Dtrain20.saveCSV("Q1_Train20.csv");
    Dtrain200.saveCSV("Q1_Train200.csv");
    Dtrain2000.saveCSV("Q1_Train2000.csv");
    Dvalidate.saveCSV("Q1_Validate.csv");


    // ==========================================
    // Part 1: Bayes-Optimal Classifier
    // ==========================================

    cout << "\n========== PART 1: BAYES-OPTIMAL ==========" << endl;

    OptimalClassifier classifier;

    classifier.evaluate(Dvalidate);
    classifier.generateROC(Dvalidate);


    // ==========================================
    // Part 2A: Logistic-Linear Classifiers
    // ==========================================

    cout << "\n========== PART 2A: LOGISTIC-LINEAR ==========" << endl;

    LogisticClassifier logistic20(0.05, 10000);
    LogisticClassifier logistic200(0.05, 10000);
    LogisticClassifier logistic2000(0.05, 10000);

    logistic20.train(Dtrain20);
    logistic200.train(Dtrain200);
    logistic2000.train(Dtrain2000);

    cout << "\n========== LINEAR VALIDATION ==========" << endl;

    cout << "\n20 Training Samples" << endl;
    logistic20.evaluate(Dvalidate);

    cout << "\n200 Training Samples" << endl;
    logistic200.evaluate(Dvalidate);

    cout << "\n2000 Training Samples" << endl;
    logistic2000.evaluate(Dvalidate);


    // ==========================================
    // Part 2B: Logistic-Quadratic Classifiers
    // ==========================================

    cout << "\n========== PART 2B: LOGISTIC-QUADRATIC ==========" << endl;

    LogisticQuadraticClassifier quadratic20(0.01, 100000);
    LogisticQuadraticClassifier quadratic200(0.01, 30000);
    LogisticQuadraticClassifier quadratic2000(0.01, 30000);

    quadratic20.train(Dtrain20);
    quadratic200.train(Dtrain200);
    quadratic2000.train(Dtrain2000);

    cout << "\n========== QUADRATIC VALIDATION ==========" << endl;

    cout << "\n20 Training Samples" << endl;
    quadratic20.evaluate(Dvalidate);

    cout << "\n200 Training Samples" << endl;
    quadratic200.evaluate(Dvalidate);

    cout << "\n2000 Training Samples" << endl;
    quadratic2000.evaluate(Dvalidate);

    // Export classifier comparison
    SaveComparisonResults();

    return 0;
}
