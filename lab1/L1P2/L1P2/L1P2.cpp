#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
using namespace std;

//multilly matrix
float** oneThreadMatrixMultiplication(int dimension, float** matrixA, float** matrixB)
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

//Matrix fullfill
float** matrixFilling(int dimension)
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

//Print matrix
void matrixPrint(int dimension, float** matrix)
{
    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
void matrixPrint(int rows, int cols, float** matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}
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

//Concatination matrix from vector
float** matrixConnection(int dimension, int stepForThread, int threadNumber, vector<float**> threadsResults)
{
    float** matrix = new float* [dimension];
    for (int i = 0; i < dimension; i++)
    {
        matrix[i] = new float[dimension];
    }

    int row = 0;
    int i = 0;
    for (auto threadResult : threadsResults)
    {
        int divisionRemains = (i == threadNumber - 1 ? fmod(dimension, threadNumber) : 0);
        for (int i = 0; i < stepForThread + divisionRemains; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                matrix[row][j] = threadResult[i][j];
            }
            row++;
        }
        i++;
    }

    return matrix;
}

//cut the row
float** matrixCutting(int dimension, int firsRow, int lastRow, float** matrix)
{
    float** cuttedMatrix = new float* [lastRow - firsRow];
    for (int i = 0; i < lastRow - firsRow; i++)
    {
        cuttedMatrix[i] = new float[dimension];
    }

    for (int i = 0; i < lastRow - firsRow; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            cuttedMatrix[i][j] = matrix[i + firsRow][j];
        }
    }

    return cuttedMatrix;
}

int main()
{
    //int dimension = 2048;
    int dimension = 4096;
    int printDimension = 4;
    int threadNumber = 8;
    int repeats = 10;

    cout << "Dimension: " << dimension << endl << "Thread number: " << threadNumber << endl << "Repeats number: " << repeats << endl;

    int stepForThread = dimension / threadNumber;

    srand(time(NULL));

    vector<float**> threadsResults(threadNumber, 0);

    auto matrixMultiplication
    {
        [&threadsResults](int threadID, int rowsA, int colsA, float** matrixA, int rowsB, int colsB, float** matrixB)
        {
            float** matrixC = new float* [rowsA];
            for (int i = 0; i < rowsA; i++)
            {
                matrixC[i] = new float[rowsB];
            }

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    float result = 0;
                    for (int k = 0; k < rowsB; k++)
                    {
                        result += matrixA[i][k] * matrixB[k][j];
                    }
                    matrixC[i][j] = result;
                }
            }
            threadsResults.at(threadID) = matrixC;
            cout << "Thread " << setw(7) << this_thread::get_id() << " is stopped calculating" << endl;
        }
    };

    vector<chrono::duration<float>> severalThreadDuration;
    vector<chrono::duration<float>> oneThreadDuration;

    float** matrixA;
    float** matrixB;
    float** matrixC;

    cout << endl;
    for (int n = 0; n < repeats; n++)
    {
        cout << endl << "-------------- Repeat " << n << " --------------" << endl;

        matrixA = matrixFilling(dimension);
        matrixB = matrixFilling(dimension);

        //multi-thread
        vector<std::thread> threads;

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < threadNumber; i++)
        {
            double divisionRemains = (i == threadNumber - 1 ? fmod(dimension, threadNumber) : 0);
            threads.emplace_back(matrixMultiplication, i, stepForThread + divisionRemains, dimension, matrixCutting(dimension, i * stepForThread, (i + 1) * stepForThread + divisionRemains, matrixA), dimension, dimension, matrixB);
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        matrixC = matrixConnection(dimension, stepForThread, threadNumber, threadsResults);

        auto end = chrono::high_resolution_clock::now();
        severalThreadDuration.emplace_back(end - start);

        matrixPrint(printDimension, matrixA, matrixB, matrixC);

        //one-thread
        start = chrono::high_resolution_clock::now();

        matrixC = oneThreadMatrixMultiplication(dimension, matrixA, matrixB);

        end = chrono::high_resolution_clock::now();
        oneThreadDuration.emplace_back(end - start);

        matrixPrint(printDimension, matrixA, matrixB, matrixC);
    }
    cout << "-----------------------------------------" << endl << endl;

    //multi-thread output
    float severalThreadDurationTime = 0;
    cout << "All several threads calculation times: " << endl;
    for (auto& time : severalThreadDuration)
    {
        cout << time.count() << endl;
        severalThreadDurationTime += time.count();
    }
    cout << "General time of several threads calculation    " << severalThreadDurationTime << endl;
    cout << "Average time of several threads calculation    " << severalThreadDurationTime / repeats << endl << endl;

    //one-thread output
    float oneThreadDurationTime = 0;
    cout << "All one thread calculation times: " << endl;
    for (auto& time : oneThreadDuration)
    {
        cout << time.count() << endl;
        oneThreadDurationTime += time.count();
    }
    cout << "General time of one thread calculation " << oneThreadDurationTime << endl;
    cout << "Average time of one thread calculation " << oneThreadDurationTime / repeats << endl << endl;

    //MEMORY CONTROL
    for (int i = 0; i < dimension; i++)
    {
        delete[] matrixA[i];
        delete[] matrixB[i];
        delete[] matrixC[i];
    }
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 1;
}