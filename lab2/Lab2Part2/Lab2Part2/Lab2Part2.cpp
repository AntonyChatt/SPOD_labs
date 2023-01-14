#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>

#include "omp.h"

using namespace std;

const int ThreadsNumber = 8;

//Mult
float** MatrixMultiplication(int dimension, float** matrixA, float** matrixB)
{
    float** matrixC = new float* [dimension];
    for (int i = 0; i < dimension; i++)
    {
        matrixC[i] = new float[dimension];
    }

    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
        {
            float result = 0;
            for (int k = 0; k < dimension; k++)
            {
                result += matrixA[i][k] * matrixB[k][j];
            }
            matrixC[i][j] = result;
        }
    return matrixC;
}
float** MatrixMultiplicationThreads(int dimension, float** matrixA, float** matrixB)
{
    float** matrixC = new float* [dimension];
#pragma omp parallel for num_threads(ThreadsNumber)
    for (int i = 0; i < dimension; i++)
    {
        matrixC[i] = new float[dimension];
    }

#pragma omp parallel for num_threads(ThreadsNumber)
    for (int i = 0; i < dimension; i++)
        for (int j = 0; j < dimension; j++)
        {
            float result = 0;
            for (int k = 0; k < dimension; k++)
            {
                result += matrixA[i][k] * matrixB[k][j];
            }
            matrixC[i][j] = result;
        }
    return matrixC;
}

//fulfil
float** matrixRandCreation(int dimension)
{
    float** matrix = new float* [dimension];
    for (int i = 0; i < dimension; i++)
    {
        matrix[i] = new float[dimension];
    }

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            matrix[i][j] = rand() % 10;
        }
    }

    return matrix;
}

//print
void matrixPrint(int dimension, float** matrixA, float** matrixB, float** matrixC)
{
    cout << endl;
    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            cout << matrixA[i][j] << " ";
        }
        cout << "   ";
        for (int j = 0; j < dimension; j++)
        {
            cout << matrixB[i][j] << " ";
        }
        cout << "   ";
        for (int j = 0; j < dimension; j++)
        {
            cout << matrixC[i][j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    int dimension = 2048;
    int printDimension = 4;
    int repeats = 5;

    cout << "Dimension: " << dimension << "\nThread number: " << ThreadsNumber << "\nRepeats number: " << repeats << endl;

    srand(time(NULL));

    vector<chrono::duration<float>> severalThreadDuration;
    vector<chrono::duration<float>> oneThreadDuration;

    float** matrixA = 0;
    float** matrixB = 0;
    float** matrixC = 0;


    for (int n = 0; n < repeats; n++)
    {
        cout << "<";

        matrixA = matrixRandCreation(dimension);
        matrixB = matrixRandCreation(dimension);

        auto start = chrono::high_resolution_clock::now();

        matrixC = MatrixMultiplicationThreads(dimension, matrixA, matrixB);

        auto end = chrono::high_resolution_clock::now();
        severalThreadDuration.emplace_back(end - start);

        //only for non random generation
        //matrixPrint(printDimension, matrixA, matrixB, matrixC);

        for (int i = 0; i < dimension; i++)
        {
            delete[] matrixA[i];
            delete[] matrixB[i];
            delete[] matrixC[i];
        }
    }
    cout << "=============================================" << endl << endl;

    //Several thread output
    float severalThreadDurationTime = 0; int i = 0;
    cout << "All several threads calculation times: ";
    for (auto& time : severalThreadDuration)
    {
        cout << endl << "Repeat " << i << ": " << time.count(); i++;
        severalThreadDurationTime += time.count();
    }
    cout << endl << "General time of several threads calculation    " << severalThreadDurationTime << endl;
    cout << "Average time of several threads calculation    " << severalThreadDurationTime / repeats << endl << endl;


    for (int n = 0; n < repeats; n++)
    {
        cout << "<";

        matrixA = matrixRandCreation(dimension);
        matrixB = matrixRandCreation(dimension);

        auto start = chrono::high_resolution_clock::now();

        matrixC = MatrixMultiplication(dimension, matrixA, matrixB);

        auto end = chrono::high_resolution_clock::now();
        oneThreadDuration.emplace_back(end - start);

        //only for non random generation
        //matrixPrint(printDimension, matrixA, matrixB, matrixC);

        for (int i = 0; i < dimension; i++)
        {
            delete[] matrixA[i];
            delete[] matrixB[i];
            delete[] matrixC[i];
        }
    }
    cout << "=============================================" << endl << endl;

    //One thread output
    float oneThreadDurationTime = 0;
    i = 0;
    cout << "All one thread calculation times: ";
    for (auto& time : oneThreadDuration)
    {
        cout << endl << "Repeat " << i << ": " << time.count(); i++;
        oneThreadDurationTime += time.count();
    }
    cout << endl << "General time of one thread calculation " << oneThreadDurationTime << endl;
    cout << "Average time of one thread calculation " << oneThreadDurationTime / repeats << endl << endl;

    // CLEAR
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 1;
}