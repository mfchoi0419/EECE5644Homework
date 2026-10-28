#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <limits>
#include <fstream>
#include <string>
#include <stdexcept>
using namespace std;

class Landmark
{
private:
    double x;
    double y;

public:
    Landmark(double xCoord, double yCoord)
    {
        x = xCoord;
        y = yCoord;
    }

    double getX() const
    {
        return x;
    }

    double getY() const
    {
        return y;
    }

    double calculateDistance(double vehicleX, double vehicleY) const
    {
        double dx = vehicleX - x;
        double dy = vehicleY - y;

        return sqrt(dx * dx + dy * dy);
    }
};

class LandmarkGenerator
{
public:
    vector<Landmark> generateLandmarks(int K)
    {
        vector<Landmark> landmarks;

        const double PI = acos(-1.0);

        for (int i = 0; i < K; i++)
        {
            double angle = 2.0 * PI * i / K;

            double x = cos(angle);
            double y = sin(angle);

            landmarks.push_back(Landmark(x, y));
        }

        return landmarks;
    }
};

class RangeMeasurementGenerator
{
private:
    mt19937 rng;
    normal_distribution<double> noiseDistribution;

public:
    RangeMeasurementGenerator(unsigned int seed = 42)
    : rng(seed), noiseDistribution(0.0, 0.3)
    {
    }

    vector<double> generateMeasurements(
        const vector<Landmark>& landmarks,
        double trueX,
        double trueY)
    {
        vector<double> measurements;

        for (int i = 0; i < static_cast<int>(landmarks.size()); i++)
        {
            double trueDistance = 
                landmarks[i].calculateDistance(trueX, trueY);

            double measuredDistance;

            do
            {
                double noise = noiseDistribution(rng);

                measuredDistance = trueDistance + noise;

            } while (measuredDistance < 0.0);

            measurements.push_back(measuredDistance);

        }

        return measurements;
    }
};

class MAPLocalizer
{
private:
    double sigmaX;
    double sigmaY;
    double sigmaNoise;

public:
    MAPLocalizer(double sx, double sy, double sn)
    {
        sigmaX = sx;
        sigmaY = sy;
        sigmaNoise = sn;
    }

    double calculateObjective(
        double x,
        double y,
        const vector<Landmark>& landmarks,
        const vector<double>& measurements) const
    {
        double objective = 0.0;

        for (int i = 0; i < static_cast<int>(landmarks.size()); i++)
        {
            double predictedDistance = 
                landmarks[i].calculateDistance(x, y);

            double difference = measurements[i] - predictedDistance;

            objective += (difference * difference) /
                        (2.0 * sigmaNoise * sigmaNoise);

        }

        double priorX = (x * x) / (2.0 * sigmaX * sigmaX);
        double priorY = (y * y) / (2.0 * sigmaY * sigmaY);

        objective += priorX + priorY;

        return objective;
    }

    void estimatePosition(
        const vector<Landmark>& landmarks,
        const vector<double>& measurements,
        double& estimatedX,
        double& estimatedY)
    {
        double minimumObjective = numeric_limits<double>::infinity();

        double step = 0.01;

        for (int i = 0; i <= 400; i++)
        {
            double x = -2.0 + i * step;

            for (int j = 0; j <= 400; j++)
            {
                double y = -2.0 + j * step;

                double objective = calculateObjective(
                    x, y, landmarks, measurements);

                if (objective < minimumObjective)
                {
                    minimumObjective = objective;

                    estimatedX = x;
                    estimatedY = y;
                }
            }
        }
    }

    void saveObjectiveGrid(
        const vector<Landmark>& landmarks,
        const vector<double>& measurements,
        const string& filename)
    {
        ofstream file(filename);

        if (!file.is_open())
        {
            cout << "Error opening file: " << filename << endl;
            return;
        }

        file << fixed << setprecision(8);

        double step = 0.02;

        for (int i = 0; i <= 200; i++)
        {
            double x = -2.0 + i * step;

            for (int j = 0; j <= 200; j++)
            {
                double y = -2.0 + j * step;

                double objective = calculateObjective(
                    x, y, landmarks, measurements);
                
                file << x << " "
                     << y << " "
                     << objective << "\n";
            }

            file << "\n";
        }

        file.close();

        cout << "Saved objective grid: " << filename << endl;
    }

    void savePlotPoints(
        const vector<Landmark>& landmarks,
        double trueX,
        double trueY,
        double estimatedX,
        double estimatedY,
        int K)
    {
        string prefix = "Q3_K" + to_string(K);

        ofstream trueFile(prefix + "_True.dat");
        trueFile << trueX << " " << trueY << endl;
        trueFile.close();

        ofstream mapFile(prefix + "_MAP.dat");
        mapFile << estimatedX << " " << estimatedY << endl;
        mapFile.close();

        ofstream landmarkFile(prefix + "_Landmarks.dat");

        for (int i = 0; i < static_cast<int>(landmarks.size()); i++)
        {
            landmarkFile << landmarks[i].getX() << " "
                         << landmarks[i].getY() << endl;
        }

        landmarkFile.close();
    }
};

int main()
{
    double trueX = 0.5;
    double trueY = 0.4;

    LandmarkGenerator generator;
    RangeMeasurementGenerator measurementGenerator(42);

    MAPLocalizer localizer(0.25, 0.25, 0.3);

    cout << fixed << setprecision(4);

    cout << "Question 3: MAP Vehicle Localization" << endl;

    cout << "\nTrue Vehicle Position: ("
         << trueX << ", " << trueY << ")" << endl;

    for (int K = 1; K <= 4; K++)
    {
        vector<Landmark> landmarks = generator.generateLandmarks(K);
        
        vector<double> measurements = 
            measurementGenerator.generateMeasurements(landmarks, trueX, trueY);

        cout << "\nK = " << K << endl;

        for (int i = 0; i < K; i++)
        {
            cout << "Landmark " << i + 1 << ": ("
                 << landmarks[i].getX() << ", "
                 << landmarks[i].getY() << ")" << endl;

            double distance = landmarks[i].calculateDistance(trueX, trueY);

            cout << "True Distance: " << distance << endl;

            cout << "Measured Distance: " << measurements[i] << endl;

            cout << "Measurement Error: " 
                 << measurements[i] - distance << endl;
        }

        double objectiveAtTrue = localizer.calculateObjective(
            trueX, trueY, landmarks, measurements);

        double objectiveAtOrigin = localizer.calculateObjective(
            0.0, 0.0, landmarks, measurements);

        cout << "\nMAP Objective at True Position: "
             << objectiveAtTrue << endl;

        cout << "MAP Objective at Origin: "
             << objectiveAtOrigin << endl;

        double estimatedX = 0.0;
        double estimatedY = 0.0;

        localizer.estimatePosition(
            landmarks, measurements, estimatedX, estimatedY);

        double localizationError = sqrt(
            pow(estimatedX - trueX, 2) +
            pow(estimatedY - trueY, 2));

        double minimumObjective = localizer.calculateObjective(
            estimatedX, estimatedY, landmarks, measurements);
        
        cout << "\nMAP Estimated Position: ("
             << estimatedX << ", " << estimatedY << ")" << endl; 

        cout << "Minimum MAP Objective: "
             << minimumObjective << endl;

        cout << "Localization Error: " 
             << localizationError << endl;

        string filename = "Q3_MAP_Grid_K" + to_string(K) + ".dat";

        localizer.saveObjectiveGrid(
            landmarks, measurements, filename);

        localizer.savePlotPoints(
            landmarks,
            trueX,
            trueY,
            estimatedX,
            estimatedY,
            K);
    }

    return 0;
}