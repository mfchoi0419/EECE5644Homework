#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
using namespace std;

class BayesianRejectClassifier
{
private:
    double lambdaR;
    double lambdaS;

public:
    BayesianRejectClassifier(double rejectLoss, double substitutionLoss)
    {
        lambdaR = rejectLoss;
        lambdaS = substitutionLoss;
    }

    int classify(const vector<double>& posterior)
    {
        if (posterior.empty() || lambdaS <= 0.0)
        {
            return -1;
        }

        int bestClass = 0;

        for (int i = 1; i < static_cast<int>(posterior.size()); i++)
        {
            if (posterior[i] > posterior[bestClass])
            {
                bestClass = i;
            }
        }

        double maximumPosterior = posterior[bestClass];

        double threshold = 1.0 - lambdaR / lambdaS;

        if (maximumPosterior >= threshold)
        {
            return bestClass;
        }

        return -1;
    }
};

int main()
{
    cout << fixed << setprecision(4);

    double lambdaR = 0.2;
    double lambdaS = 1.0;

    BayesianRejectClassifier classifier(lambdaR, lambdaS);

    vector<vector<double>> testSamples =
    {
        {0.90, 0.05, 0.05},
        {0.60, 0.30, 0.10},
        {0.20, 0.75, 0.05},
        {0.34, 0.33, 0.33}
    };

    cout << "Question 4: Bayesian Classification with Rejection" << endl;

    cout << "\nRejection Loss: " << lambdaR << endl;
    cout << "Substitution Loss: " << lambdaS << endl;

    cout << "Decision Threshold: "
         << 1.0 - lambdaR / lambdaS << endl;

    for (int i = 0; i < static_cast<int>(testSamples.size()); i++)
    {
        int decision = classifier.classify(testSamples[i]);

        cout << "\nSample " << i + 1 << endl;

        cout << "Posterior Probabilities: ";

        for (int j = 0; j < static_cast<int>(testSamples[i].size()); j++)
        {
            cout << testSamples[i][j] << " ";
        }

        if (decision == -1)
        {
            cout << "\nDecision: REJECT" << endl;
        }
        else
        {
            cout << "\nDecision: Class " << decision + 1 << endl;
        }
    }

    cout << "\nSpecial Case 1: Zero Rejection Loss" << endl;

    BayesianRejectClassifier zeroRejectClassifier(0.0, 1.0);

    cout << "Decision Threshold: " << 1.0 << endl;

    for (int i = 0; i < static_cast<int>(testSamples.size()); i++)
    {
        int decision = zeroRejectClassifier.classify(testSamples[i]);

        cout << "Sample " << i + 1 << ": ";

        if (decision == -1)
        {
            cout << "REJECT" << endl;
        }
        else
        {
            cout << "Class " << decision + 1 << endl;
        }
    }

    cout << "\nSpecial Case 2: Rejection Loss > Substitution Loss" << endl;

    double highRejectLoss = 1.5;
    double substitutionLoss = 1.0;

    BayesianRejectClassifier highRejectClassifier(
        highRejectLoss, substitutionLoss);

    cout << "Decision Threshold: "
        << 1.0 - highRejectLoss / substitutionLoss << endl;

    for (int i = 0; i < static_cast<int>(testSamples.size()); i++)
    {
        int decision = highRejectClassifier.classify(testSamples[i]);

        cout << "Sample " << i + 1 << ": ";

        if (decision == -1)
        {
            cout << "REJECT" << endl;
        }
        else
        {
            cout << "Class " << decision + 1 << endl;
        }
    }

    return 0;
}