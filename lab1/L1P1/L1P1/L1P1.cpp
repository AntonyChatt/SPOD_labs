#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
using namespace std;

// series member (any)
double seriesMember(double n)
{
	return n;
}

//one-thread summ calc
double oneThreadSeriesCalculation(double seriesLowerLimit, double seriesUpperLimit)
{
	double result = 0;
	for (double i = seriesLowerLimit; i < seriesUpperLimit; i++)
	{
		result += seriesMember(i);
	}
	cout << "Thread " << setw(7) << this_thread::get_id() << " is stopped calculating" << endl;
	return result;
}


int main()
{
	cout << "Lab 1 Task 1" << endl;

	double seriesUpperLimit = 10000000000;
	double seriesLowerLimit = 1;
	int threadNumber = 8;
	int repeats = 10;

	cout << "Series: " << seriesUpperLimit << endl << "Thread number: " << threadNumber << endl << "Repeats number: " << repeats << endl;

	vector<double> threadsResults(threadNumber, 0.);
	
	//Calc summ
	auto seriesCalculation
	{
		[&threadsResults](int threadID, double seriesLowerLimit, double seriesUpperLimit)
		{
			double result = 0;
			for (double i = seriesLowerLimit; i < seriesUpperLimit; i++)
			{
				result += seriesMember(i);
			}
			threadsResults.at(threadID) = result;
			cout << "Thread " << setw(7) << this_thread::get_id() << " is stopped calculating" << endl;
		}
	};

	//Calc step
	double stepForThread = seriesUpperLimit / threadNumber;

	vector<chrono::duration<float>> severalThreadDuration;
	vector<chrono::duration<float>> oneThreadDuration;

	for (int n = 0; n < repeats; n++)
	{
		cout << endl << "---------------- Repeat " << n << " ------------------" << endl;

		vector<std::thread> threads;

		auto start = chrono::high_resolution_clock::now();

		for (int i = 0; i < threadNumber; i++)
		{
			double divisionRemains = (i == threadNumber - 1 ? fmod(seriesUpperLimit, threadNumber) : 0);
			threads.emplace_back(seriesCalculation, i, i * stepForThread, (i + 1) * stepForThread + divisionRemains);
		}

		for (auto& thread : threads)
		{
			thread.join();
		}

		auto end = chrono::high_resolution_clock::now();
		severalThreadDuration.emplace_back(end - start);

		cout << "Result of several threads calculation:    " << std::accumulate(threadsResults.begin(), threadsResults.end(), 0.) << endl;
		
		start = chrono::high_resolution_clock::now();

		double oneThreadResult = oneThreadSeriesCalculation(seriesLowerLimit, seriesUpperLimit);

		end = chrono::high_resolution_clock::now();
		oneThreadDuration.emplace_back(end - start);

		cout << "Result of one thread calculation: " << oneThreadResult << endl;
	}

	cout << "-----------------------------------------" << endl << endl;
	
	//Output multi-thread calc results
	float severalThreadDurationTime = 0;
	cout << "All several threads calculation times: " << endl;
	for (auto& time : severalThreadDuration)
	{
		cout << time.count() << endl;
		severalThreadDurationTime += time.count();
	}
	cout << "General time of several threads calculation    " << severalThreadDurationTime << endl;
	cout << "Average time of several threads calculation    " << severalThreadDurationTime / repeats << endl << endl;

	//Output one-thread calc results
	float oneThreadDurationTime = 0;
	cout << "All one thread calculation times: " << endl;
	for (auto& time : oneThreadDuration)
	{
		cout << time.count() << endl;
		oneThreadDurationTime += time.count();
	}
	cout << "General time of one thread calculation " << oneThreadDurationTime << endl;
	cout << "Average time of one thread calculation " << oneThreadDurationTime / repeats << endl << endl;

	return 1;
}