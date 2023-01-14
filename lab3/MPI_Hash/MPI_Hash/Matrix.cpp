#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#include "mpi.h"
#include "md5.h"

#define size 4096

int main() {
    auto A      = new float[size][size];
    auto B      = new float[size][size];
    auto result = new float[size][size]{NULL};

    int nIterations = 10;
    std::vector<long long> timeMultipleThread(nIterations);
    std::vector<long long> timeSingleThread(nIterations);

    for (int i = 0; i < nIterations; ++i) {
        std::cout << "\nIteration #" << i << std::endl;
        auto result = new float[size][size];
        auto resultSingleThread = new float[size][size];

        int tag = 0;
        int nProcesses, rank;

        auto start = std::chrono::system_clock::now();

        MPI_Init(NULL, NULL);
        MPI_Status status;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Comm_size(comm, &nProcesses);
        MPI_Comm_rank(comm, &rank);

        if (rank == 0) {
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<> getFloat(-RAND_MAX, RAND_MAX);

            auto createFloats{
                    [&A, &B, &getFloat, &eng](unsigned int from, unsigned int to) {
                        for (unsigned int i = from; i < to; ++i) {
                            for (unsigned int j = 0; j < size; ++j) {
                                A[i][j] = (float)getFloat(eng);
                                B[i][j] = (float)getFloat(eng);
                            }
                        }
                    }
            };

            std::vector<std::thread> threadVectorRandomizer;
            unsigned int nThreads = std::thread::hardware_concurrency();
            unsigned int step = size / nThreads;

            for (unsigned int k = 0; k < nThreads; ++k) {
                unsigned int remains = (k == nThreads - 1) ? size % nThreads : 0;
                threadVectorRandomizer.emplace_back(createFloats, k * step, (k + 1) * step + remains);
            }

            for (auto& thread : threadVectorRandomizer) {
                thread.join();
            }

            step = size / (nProcesses == 1 ? 1 : (nProcesses - 1));

            for (int i = 1; i < nProcesses; i++) {
                int remains = (i == nProcesses - 1) ? (int)size % nProcesses : 0;
                int from = i * step;
                int to = (i + 1) * step + remains;

                MPI_Send(&A,    size * size, MPI_FLOAT, i, tag,     comm); // A
                MPI_Send(&B,    size * size, MPI_FLOAT, i, tag + 1, comm); // A
                MPI_Send(&from, 4,           MPI_INT,   i, tag + 2, comm); // from
                MPI_Send(&to,   4,           MPI_INT,   i, tag + 3, comm); // to
            }

            for (int i = 1; i < nProcesses; i++) {
                auto tempResult = new float[size][size]{ NULL };
                int remains = (i == nProcesses - 1) ? (int)size % nProcesses : 0;
                int from = i * step;
                int to = (i + 1) * step + remains;

                MPI_Recv(&tempResult, size * size, MPI_FLOAT, i, tag, comm, &status); // temp result

                for (size_t i = from; i < to; i++) {
                    for (size_t j = 0; i < size; j++) {
                        result[i][j] = tempResult[i][j];
                    }
                }
            }

            auto end = std::chrono::system_clock::now();
            auto elapsed_nanoseconds = (end - start).count();
            std::cout << "MPI-8 time spent, ns: " << elapsed_nanoseconds << std::endl;
            timeMultipleThread.at(i) = elapsed_nanoseconds;

            start = std::chrono::system_clock::now();
            auto resultSingleThread = new float[size][size];

            for (unsigned int i = 0; i < size; ++i) {
                for (unsigned int j = 0; j < size; ++j) {
                    resultSingleThread[i][j] = 0;

                    for (unsigned int j2 = 0; j2 < size; ++j2) {
                        resultSingleThread[i][j] += A[i][j2] * B[j2][j];
                    }
                }
            }

            end = std::chrono::system_clock::now();
            elapsed_nanoseconds = (end - start).count();
            std::cout << "1 th. time spent, ns: " << elapsed_nanoseconds << std::endl;
            timeSingleThread.at(i) = elapsed_nanoseconds;
        } else {
            int flag = 1;
            unsigned int from, to;

            MPI_Iprobe(0, tag,     comm, &flag, &status);
            MPI_Iprobe(0, tag + 1, comm, &flag, &status);
            MPI_Iprobe(0, tag + 2, comm, &flag, &status);
            MPI_Iprobe(0, tag + 3, comm, &flag, &status);
            std::cout << rank << "   probe   success" << std::endl;
            MPI_Recv(&A,    size * size, MPI_FLOAT, 0, tag,     comm, &status); // A
            MPI_Recv(&B,    size * size, MPI_FLOAT, 0, tag + 1, comm, &status); // B
            MPI_Recv(&from, 4,           MPI_INT,   0, tag + 2, comm, &status); // from
            MPI_Recv(&to,   4,           MPI_INT,   0, tag + 3, comm, &status); // to

            for (unsigned int i = from; i < to; ++i) {
                for (unsigned int j = 0; j < size; ++j) {
                    result[i][j] = 0;

                    for (unsigned int j2 = 0; j2 < size; ++j2) {
                        result[i][j] += A[i][j2] * B[j2][j];
                    }
                }
            }

            MPI_Send(&result, size * size, MPI_FLOAT, 0, tag, comm); // result
            std::cout << rank << " computed" << std::endl;
        }

        MPI_Finalize();
    }
}