#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

class CategoricalDataset
{
private:
    int K;
    vector<vector<int>> samples;

public:
    CategoricalDataset(int numCategories)
    {
        K = numCategories;
    }

    void addSample(int category)
    {
        if (category < 0 || category >= K)
        {
            cout << "Invalid category" << endl;
            return;
        }

        vector<int> sample(K, 0);

        sample[category] = 1;

        samples.push_back(sample);
    }

    int getSize() const
    {
        return static_cast<int>(samples.size());
    }

    int getCategories() const
    {
        return K;
    }

    vector<int> getCounts() const
    {
        vector<int> counts(K, 0);

        for (int i = 0; i < static_cast<int>(samples.size()); i++)
        {
            for (int j = 0; j < K; j++)
            {
                counts[j] += samples[i][j];
            }
        }

        return counts;
    }

    void printSamples() const
    {
        for (int i = 0; i < static_cast<int>(samples.size()); i++)
        {
            cout << "Sample " << i + 1 << ": [";

            for (int j = 0; j < K; j++)
            {
                cout << samples[i][j];

                if (j < K - 1)
                {
                    cout << ", ";
                }
            }

            cout << "]" << endl;
        }
    }
};

class CategoricalEstimator
{
public:
    vector<double> estimateML(const CategoricalDataset& dataset)
    {
        vector<int> counts = dataset.getCounts();

        int N = dataset.getSize();
        int K = dataset.getCategories();

        vector<double> theta(K, 0.0);

        if (N == 0)
        {
            cout << "Error: Dataset is empty" << endl;
            return theta;
        }

        for (int i = 0; i < K; i++)
        {
            theta[i] = static_cast<double>(counts[i]) / N;
        }

        return theta;
    }

    vector<double> estimateMAP(
        const CategoricalDataset& dataset,
        const vector<double>& alpha)
    {
        int N = dataset.getSize();
        int K = dataset.getCategories();

        vector<double> theta(K, 0.0);
        vector<int> counts = dataset.getCounts();

        if (static_cast<int>(alpha.size()) != K)
        {
            cout << "Error: Alpha size does not match K" << endl;
            return theta;
        }

        double denominator = static_cast<double>(N);

        for (int i = 0; i < K; i++)
        {
            if (alpha[i] <= 0.0)
            {
                cout << "Error: Alpha must be positive" << endl;
                return theta;
            }

            if (counts[i] + alpha[i] <= 1.0)
            {
                cout << "Error: Posterior parameter must exceed 1" << endl;
                return theta;
            }

            denominator += alpha[i] - 1.0;
        }

        for (int i = 0; i < K; i++)
        {
            theta[i] = (counts[i] + alpha[i] - 1.0) / denominator;
        }

        return theta;
    }
};

int main()
{
    cout << fixed << setprecision(4);

    cout << "Question 5: Categorical Distribution Estimation" << endl;

    CategoricalDataset dataset(3);

    dataset.addSample(0);
    dataset.addSample(1);
    dataset.addSample(0);
    dataset.addSample(2);
    dataset.addSample(1);
    dataset.addSample(0);
    dataset.addSample(0);
    dataset.addSample(2);
    dataset.addSample(1);
    dataset.addSample(0);

    cout << "\nNumber of Categories: "
         << dataset.getCategories() << endl;

    cout << "Number of Samples: " 
         << dataset.getSize() << endl;

    cout << "\nOne-Hot Encoded Samples:" << endl;

    dataset.printSamples();

    vector<int> counts = dataset.getCounts();

    cout << "\nCategory Counts:" << endl;

    for (int i = 0; i < dataset.getCategories(); i++)
    {
        cout << "Category " << i + 1 << ": "
             << counts[i] << endl;
    }

    CategoricalEstimator estimator;

    vector<double> thetaML = estimator.estimateML(dataset);

    cout << "\nMaximum Likelihood Estimates:" << endl;

    double sumML = 0.0;

    for (int i = 0; i < dataset.getCategories(); i++)
    {
        cout << "Theta " << i + 1 << ": "
             << thetaML[i] << endl;

        sumML += thetaML[i];
    }

    cout << "Sum of ML Estimates: " << sumML << endl;

    vector<double> alpha = {2.0, 2.0, 2.0};

    vector<double> thetaMAP = estimator.estimateMAP(dataset, alpha);

    cout << "\nMaximum A Posteriori Estimates:" << endl;

    double sumMAP = 0.0;

    for (int i = 0; i < dataset.getCategories(); i++)
    {
        cout << "Theta " << i + 1 << ": "
             << thetaMAP[i] << endl;

        sumMAP += thetaMAP[i];
    }

    cout << "Sum of MAP Estimates: " << sumMAP << endl;

    cout << "\nComparing Different Dirichlet Priors:" << endl;

    vector<vector<double>> alphaTests = 
    {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {5.0, 5.0, 5.0},
        {10.0, 10.0, 10.0},
        {5.0, 2.0, 1.0}
    };

    for (int i = 0; i < static_cast<int>(alphaTests.size()); i++)
    {
        vector<double> estimates = 
            estimator.estimateMAP(dataset, alphaTests[i]);

        cout << "\nAlpha = [";

        for (int j = 0; j < dataset.getCategories(); j++)
        {
            cout << alphaTests[i][j];

            if (j < dataset.getCategories() - 1)
            {
                cout << ", ";
            }
        }

        cout << "]" << endl;

        for (int j = 0; j < dataset.getCategories(); j++)
        {
            cout << "Theta " << j + 1 << ": "
                 << estimates[j] << endl;
        }
    }

    return 0;
}