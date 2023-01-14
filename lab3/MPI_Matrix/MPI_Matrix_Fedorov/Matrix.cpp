#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#include "mpi.h"
constexpr int sizeM = 2048;
using namespace std;



	int main() {
		auto A = new float[sizeM][sizeM]{};
		auto B = new float[sizeM][sizeM]{};
		int tag = 0;
		int processes, rank;
		MPI_Init(NULL, NULL);
		MPI_Status status;
		MPI_Comm comm = MPI_COMM_WORLD;
		MPI_Comm_size(comm, &processes);
		MPI_Comm_rank(comm, &rank);

		if (rank == 0) {
			random_device rd;
			default_random_engine eng(rd());
			uniform_real_distribution<> getfloat(-RAND_MAX, RAND_MAX);

			auto createFloates{
					[&A, &B, &getfloat, &eng](unsigned int from, unsigned int to) {
						for (unsigned int i = from; i < to; ++i) {
							for (unsigned int j = 0; j < sizeM; ++j) {
								A[i][j] = (float)getfloat(eng) / 10000;
								B[i][j] = (float)getfloat(eng) / 10000;
							}
						}
					}
			};

			vector<thread> threadVectorRandomizer;
			unsigned int nThreads = thread::hardware_concurrency();
			unsigned int step = sizeM / nThreads;

			for (unsigned int k = 0; k < nThreads; ++k) {
				unsigned int remains = (k == nThreads - 1) ? sizeM % nThreads : 0;

				threadVectorRandomizer.emplace_back(createFloates, k * step, (k + 1) * step + remains);
			}

			for (auto& thread : threadVectorRandomizer) {
				thread.join();
			}

			cout << "A and B fulfilled startl" << endl;
			step = sizeM / (processes == 1 ? 1 : (processes - 1));
			auto start = chrono::system_clock::now();

			for (int i = 1; i < processes; i++) {
				int remains = (i == processes - 1) ? (int)sizeM % (processes - 1) : 0;
				int fromIndex = (i - 1) * step;
				int toIndex = i * step + remains;
				MPI_Send(&(A[0][0]), sizeM * sizeM, MPI_FLOAT, i, tag, comm);
				MPI_Send(&(B[0][0]), sizeM * sizeM, MPI_FLOAT, i, tag + 1, comm);
				MPI_Send(&fromIndex, 4, MPI_INT, i, tag + 2, comm);
				MPI_Send(&toIndex, 4, MPI_INT, i, tag + 3, comm);
			}

			auto result = new float[sizeM][sizeM]{};

			for (int i = 1; i < processes; i++) {
				int remains = (i == processes - 1) ? (int)sizeM % (processes - 1) : 0;
				int fromIndex = (i - 1) * step;
				int toIndex = i * step + remains;
				int recvSize;

				// portion
				auto tempResult = new float[toIndex - fromIndex][sizeM]{};
				MPI_Probe(i, tag, comm, &status);
				MPI_Recv(&(tempResult[0][0]), (toIndex - fromIndex) * sizeM, MPI_FLOAT, i, tag, comm, &status);
				MPI_Get_count(&status, MPI_FLOAT, &recvSize);

				for (int i1 = fromIndex; i1 < toIndex; i1++) {
					for (int j = 0; j < sizeM; j++) {
						result[i1][j] = tempResult[i1 - fromIndex][j];
					}
				}
			}

			auto end = chrono::system_clock::now();
			auto elapsed_nanoseconds = (end - start).count();
			cout << "Several thread time: " << elapsed_nanoseconds / 1000 << "s" << endl;

			start = chrono::system_clock::now();
			auto resultSingleThread = new float[sizeM][sizeM]{};

			for (size_t i2 = 0; i2 < sizeM; ++i2) {
				for (size_t j = 0; j < sizeM; ++j) {
					resultSingleThread[i2][j] = 0.0;

					for (size_t j2 = 0; j2 < sizeM; ++j2) {
						resultSingleThread[i2][j] += A[i2][j2] * B[j2][j];
					}
				}
			}

			end = chrono::system_clock::now();
			elapsed_nanoseconds = (end - start).count();
			cout << "One thread time: " << elapsed_nanoseconds / 1000 << "s" << endl;

			for (int r1 = 0; r1 < sizeM; ++r1) {
				for (int r2 = 0; r2 < sizeM; ++r2) {
					if (result[r1][r2] != resultSingleThread[r1][r2]) {
						cerr << "result8[" << r1 << "][" << r2 << "] = " << result[r1][r2] << "  != " << " result1[" << r1 << "][" << r2 << "] = " << resultSingleThread[r1][r2] << endl;
						cerr << "The mpi matrix may be filled with 0s: " << result[140][80] << endl;
						r1 = sizeM - 1;
						break;
					}
				}
			}
		} else {
			int flag = 0;
			int fromIndex;
			int toIndex;
			MPI_Iprobe(0, tag, comm, &flag, &status);
			MPI_Iprobe(0, tag + 1, comm, &flag, &status);
			MPI_Iprobe(0, tag + 2, comm, &flag, &status);
			MPI_Iprobe(0, tag + 3, comm, &flag, &status);
			MPI_Recv(&(A[0][0]), sizeM * sizeM, MPI_FLOAT, 0, tag, comm, &status);
			MPI_Recv(&(B[0][0]), sizeM * sizeM, MPI_FLOAT, 0, tag + 1, comm, &status);
			MPI_Recv(&fromIndex, 4, MPI_INT, 0, tag + 2, comm, &status);
			MPI_Recv(&toIndex, 4, MPI_INT, 0, tag + 3, comm, &status);
			auto tempResult = new float[toIndex - fromIndex][sizeM]{};

			for (int i = fromIndex; i < toIndex; ++i) {
				for (int j = 0; j < sizeM; ++j) {
					tempResult[i - fromIndex][j] = 0.0;

					for (int j2 = 0; j2 < sizeM; ++j2) {
						tempResult[i - fromIndex][j] += A[i][j2] * B[j2][j];
					}
				}
			}

			MPI_Send(&(tempResult[0][0]), (toIndex - fromIndex) * sizeM, MPI_FLOAT, 0, tag, comm);
			cout << rank << " finished" <<endl;
		}

		cout << rank << " died" << endl;
		MPI_Finalize();
	}