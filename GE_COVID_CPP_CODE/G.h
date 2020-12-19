#ifndef _G_H_
#define _G_H_

#include "head.h"

class G {
public:
	//main
	void initialize(int, char**, Files& files, Params& params, Lists& lists, Network& network, GSL& gsl);
	void runSimulation(Params&, Lists&, Counts&, Network&, GSL&, Files&);
	//spread
	void count_states(Params&, Counts&, Network&);
	void init(Params&, Counts&, Network&, GSL&, Lists&, Files&);
	void seedInfection(Params&, Counts&, Network&, GSL&, Lists&, Files&);
	void spread(int, Files&, Lists&, Network&, Params&, GSL&, Counts&);
	void infection(Lists&, Network&, Params&, GSL&, Counts&);
	void infect(int, int, Network&, Params&, GSL&, Lists&, Counts&);
	void latency(Params& par, Lists&, GSL&, Network&, Counts&);
	void IaToR();
	void preToI();
	void IsTransition(Params&, Lists&, Network&, Counts&, GSL&);
	void homeTransition();
	void hospitals();
	void updateTime(Files&);
	void resetVariables(Lists&, Files&);
	void resetNodes(Params&, Network&);
	void resetNew(Lists&);
	void updateLists(Lists&, Network&);
	void results(int, Lists&, Files&);
	//read
	void readData(Params&, Lists& lists, Network&, Files&);
	void readParameters(char* filenm, Params&);
	void readNetwork(Params&, Lists&, Network&, Files&);
	void readNodes(Params&, Files&, Network&);
	//utilities
	void allocateMemory(Params&, Lists&);
	void initRandom(int, GSL&);
	void openFiles(Files&);
	void setBeta(Params&);
	void addToList(List*, int);
	int removeFromList(List*, int);
	void updateList(List*, List*, Network&);
	void closeFiles(Files&);
	void print(int);
	void freeMemory(Params& p, Network& network, Lists& l, GSL&);
};
	
#endif
