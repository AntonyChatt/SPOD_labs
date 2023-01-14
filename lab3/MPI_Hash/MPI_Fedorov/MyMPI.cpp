#include <stdio.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <string>

#include "mpi.h"
#include "md5.h"

using namespace std;

string tryMatch(const string hash, int step, int remains, string allowedChars, int iteration)
{
    size_t startIndex = static_cast<size_t>(step) * iteration;
    size_t endIndex = startIndex + step + remains;
    size_t length = allowedChars.length();

    if (endIndex > length) endIndex = length;
    string result;

    for (size_t i = startIndex; i < endIndex; i++)
    {
        for (size_t j = 0; j < length; j++)
        {
            for (size_t k = 0; k < length; k++)
            {
                for (size_t l = 0; l < length; l++)
                {
                    result = "";

                    result.push_back((char)allowedChars.at(i));
                    result.push_back((char)allowedChars.at(j));
                    result.push_back((char)allowedChars.at(k));
                    result.push_back((char)allowedChars.at(l));
                    string anotherHash = md5(result);

                    if (hash.compare(anotherHash) == 0)
                    {
                        return result;
                    }
                }
            }
        }
    }

    return "-1";
}

int main() {
    int tag = 0;
    int nProcesses, rank;
    string alphabet("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890");
    size_t allowedCharsCount = alphabet.length();

    int passwordLength = 4;
    string password;

    auto start = chrono::high_resolution_clock::now();

    MPI_Init(NULL, NULL);

    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nProcesses);

    //Start algorytm
    char buf[34];
    char pwd[5];

    if (rank == 0) {
        srand(time(NULL));

        // generate random password
        for (int i = 0; i < passwordLength; ++i) {
            password += alphabet.at(rand() % ((int)alphabet.length() + 1));
        }

        //generate hash
        const string md5hash = md5(password);
        cout << password << " - " << md5hash << endl;

        for (int i = 1; i < nProcesses; i++) {
            string hashIteration = md5hash;
            hashIteration += to_string(i);

            strcpy_s(buf, hashIteration.c_str());
            MPI_Send(buf, (int)hashIteration.size() + 1, MPI_CHAR, i, tag, comm);
        }

        MPI_Recv(pwd, passwordLength + 1, MPI_CHAR, MPI_ANY_SOURCE, tag, comm, &status);
        cout << "result: ";
        string result = string(pwd);
        cout << result << endl;

        if (password.compare(result) == 0) {
            cout << "Result: " << result << endl;

            auto end = chrono::high_resolution_clock::now();
            auto processTime = (end - start).count();
            cout << "Time: " << processTime << endl;
        }
        else {
            cout << "Error" << endl;
        }

        MPI_Abort(comm, 1);
    }
    else
    {
        size_t step = allowedCharsCount / (nProcesses == 1 ? 1 : (static_cast<unsigned long long>(nProcesses) - 1));
        int recvSize;
        int flag = 1;

        MPI_Iprobe(0, tag, comm, &flag, &status);
        MPI_Get_count(&status, MPI_CHAR, &recvSize);

        MPI_Recv(buf, 40, MPI_CHAR, 0, tag, comm, &status);
        string hash = "";

        for (size_t i = 0; i < 32; i++) {
            hash += buf[i];
        }

        int iteration = atoi(&buf[32]);
        int remains = (rank == nProcesses - 1) ? (int)allowedCharsCount % nProcesses : 0;

        string passwordToSend = tryMatch(hash, (int)step, remains, alphabet, iteration - 1);

        if (passwordToSend.compare("-1") != 0) {
            strcpy_s(pwd, passwordToSend.c_str());
            pwd[4] = 0;

            MPI_Send(pwd, passwordLength + 1, MPI_CHAR, 0, tag, comm);
            cout << rank << " sent \"" << passwordToSend.c_str() << "\"" << endl;
        }
    }
    cout << rank << " enden" << endl;
    MPI_Finalize();
}


