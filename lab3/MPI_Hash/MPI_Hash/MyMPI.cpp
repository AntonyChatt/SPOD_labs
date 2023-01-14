#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>
#include <string>

#include "mpi.h"
#include "md5.h"

using namespace std;

int main() {
    int tag = 0;
    int nProcesses, rank;
    std::string abc("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890");
    size_t allowedCharsCount = abc.length();
    // password length
    int sequenceLength = 4;
    std::string password;
    auto start = std::chrono::system_clock::now();
    MPI_Init(NULL, NULL);
    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nProcesses);
    MPI_Comm_rank(comm, &rank);
    char buf[34];
    char pwd[5];

    if (rank == 0) {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<> getRandomAbcSymbol(0, (int)abc.length());

        // init random password
        for (int i = 0; i < sequenceLength; ++i) {
            password += abc.at((int)getRandomAbcSymbol(eng));
        }

        // and its hash
        const std::string md5hash = md5(password);
        std::cout << password << " - " << md5hash << std::endl;

        for (int i = 1; i < nProcesses; i++) {
            std::string hashIteration = md5hash;
            // Add i to differ the current iteration for:
            // 1. dividing abc
            // 2. start hash matching from particular symbol
            hashIteration += std::to_string(i);
            // use root's buf to transfer
            strcpy_s(buf, hashIteration.c_str());
            MPI_Send(buf, (int) hashIteration.size() + 1, MPI_CHAR, i, tag, comm);
        }

        // root (0) will recieve password once it's found wherever
        MPI_Recv(pwd, sequenceLength + 1, MPI_CHAR, MPI_ANY_SOURCE, tag, comm, &status);
        std::cout << "result: ";
        // stringifying
        std::string result = std::string(pwd);
        std::cout << result << std::endl;

        // we must succeed
        if (password.compare(result) == 0) {
            std::cout << "Result: " << result << std::endl;
            auto end = std::chrono::system_clock::now();
            auto elapsed_nanoseconds = (end - start).count();
            std::cout << "Time, ns: " << elapsed_nanoseconds << std::endl;
        } else {
            std::cerr << "Error" << std::endl;
        }

        // we didn't succeed
        MPI_Abort(comm, 1);
    } else {
        // defining cetain portions for all processes
        size_t step = allowedCharsCount / (nProcesses == 1 ? 1 : (static_cast<unsigned long long>(nProcesses) - 1));
        int recvSize;
        int flag = 1;
        MPI_Iprobe(0, tag, comm, &flag, &status);
        MPI_Get_count(&status, MPI_CHAR, &recvSize);
        // using buf to recv hash + iteration number
        MPI_Recv(buf, 40, MPI_CHAR, 0, tag, comm, &status);
        std::string hash = "";

        // leave the last, separating the other part (hash)
        for (size_t i = 0; i < 32; i++) {
            hash += buf[i];
        }

        // recved current iteration
        int iteration = atoi(&buf[32]);
        int remains = (rank == nProcesses - 1) ? (int) allowedCharsCount % nProcesses : 0;
        // start matching
        std::string passwordToSend = tryMatch(hash, (int)step, remains, abc, iteration - 1);

        // if we succeeded on this process
        if (passwordToSend.compare("-1") != 0) {
            strcpy_s(pwd, passwordToSend.c_str());
            pwd[4] = 0;
            MPI_Send(pwd, sequenceLength + 1, MPI_CHAR, 0, tag, comm);
            std::cout << rank << " sent \"" << passwordToSend.c_str() << "\"" << std::endl;
        }
    }

    std::cout << rank << " died" << std::endl;
    MPI_Finalize();
}

std::string tryMatch(const std::string hash, int step, int remains, std::string allowedChars, int iteration) {

    // specifying the abc starter symbol
    size_t startIndex = static_cast<size_t>(step) * iteration;
    // and this
    size_t endIndex = startIndex + step + remains;
    // and this
    size_t length = allowedChars.length();
    // just in case
    if (endIndex > length) endIndex = length;
    std::string result;

    for (size_t i = startIndex; i < endIndex; i++) {
        for (size_t j = 0; j < length; j++) {
            for (size_t k = 0; k < length; k++) {
                for (size_t l = 0; l < length; l++) {
                    result = "";
                    // primarily constructing new 4-symbol sequence
                    result.push_back((char)allowedChars.at(i));
                    result.push_back((char)allowedChars.at(j));
                    result.push_back((char)allowedChars.at(k));
                    result.push_back((char)allowedChars.at(l));
                    std::string anotherHash = md5(result);

                    // check if we succeeded
                    if (hash.compare(anotherHash) == 0) {
                        return result;
                    }
                }
            }
        }
    }

    return "-1";
}
