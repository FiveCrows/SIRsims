#ifndef _G_H_
#define _G_H_

#include <vector>
#include "head.h"

class G {
public:
	Params& par;
	Files& files;
	Lists& lists;
	Network& net;
	Counts& counts;
	GSL& gsl;

public:
	// Constructor
	G(Params&, Files&, Lists&, Counts&, Network&, GSL&);
	//main
	void initialize(int, char**, Files& files, Params& params, Lists& lists, Network& network, GSL& gsl);
	void runSimulation(Params&, Lists&, Counts&, Network&, GSL&, Files&);
	//spread
	void countStates(Params&, Counts&, Network&, Files&);
    void writeStates(Counts&);
	void init(Params&, Counts&, Network&, GSL&, Lists&, Files&);
	void seedInfection(Params&, Counts&, Network&, GSL&, Lists&, Files&);
	void spread(int, Files&, Lists&, Network&, Params&, GSL&, Counts&);
    void vaccinations(Params&, Lists&, GSL&, Network&, Counts&, float cur_time);
    void secondVaccination(Params& par, Lists& l, GSL& gsl, Network &net, Counts& c, float cur_time);
	void vaccinateNextBatch(Network&, Lists&, Counts&, Params&, GSL&, int n, float cur_time);
    void vaccinateAquaintance(Network&, Lists&, Counts&, Params&, GSL&, int, float cur_time);
	void infection(Lists&, Network&, Params&, GSL&, Counts&, float cur_time);
	void infect(int, int, Network&, Params&, GSL&, Lists&, Counts&, float cur_time);
	void latency(Params& par, Lists&, GSL&, Network&, Counts&, float cur_time);
	void IaToR();
	void preToI();
	void IsTransition(Params&, Lists&, Network&, Counts&, GSL&, float cur_time);
	void homeTransition();
	void hospitals();
	void updateTime();
	void resetVariables(Lists&, Files&, Params& p);
	void resetNodes(Params&, Network&);
	void resetNew(Lists&);
	void updateLists(Lists&, Network&);
	void printResults(int, Lists&, Files&);
	//read
	void readData(Params&, Lists& lists, Network&, Files&);
	void readParameters(char* filenm, Params&);
	void readNetwork(Params&, Lists&, Network&, Files&);
	void readNodes(Params&, Files&, Network&);
	void readVaccinations(Params&, Files&, Network&, Lists&);
	//utilities
	void allocateMemory(Params&, Lists&);
	void initRandom(GSL&);
	void openFiles(Files&);
	void setBeta(Params&);
	void addToList(List*, int);
	int removeFromList(List*, int);
	void updateList(List*, List*, Network&);
	void closeFiles(Files&);
	void print(int);
	void freeMemory(Params& p, Network& network, Lists& l, GSL&);
	void printTransitionStats();
	void stateTransition(int source, int target, int from_state, int to_state, float from_time, float to_time);
	float readInt(FILE*);
	float readFloat(FILE*);
	void parse(int, char**, Params&);
    double getBetaISt(Node&);
	void weibull(std::vector<double>& samples, GSL&, double shape, double scale, int n);
	void lognormal(std::vector<double>& samples, GSL& gsl, double mu, double sigma2, int n);
};
	
#endif
