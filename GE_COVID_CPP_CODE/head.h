#ifndef  _HEAD_H_
#define  _HEAD_H_

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <vector>

#define NAGE 5
#define NCOMPARTMENTS 9

// Nodes refer to people in no particular order
typedef struct Node {
  //Info
  int age;
  //Infection
  int state;
  int hospitalization;
  //Connections
  int k;
  int *v;
  float *w;
  int is_vacc;      // 0,1 whether vaccinated or not

  // individual transmissibility (could be a function of time)
  // Leaky Vaccine (partially effective all all vaccinated people)
  // Hard Vaccine: Those vaccinated are immediately R. (equivalent to beta=infinity)
  // Assume that the vaccine is immediately effective
  // beta_is = params.beta * (1.-vacc1_effectiveness)
  // beta_is = params.beta * (1.-vacc2_effectiveness)
  float beta_IS;    
  float mu; // individual recovery

  float vacc_infect;  // Vaccine doses reduce infectiousness of others
  float vacc_suscept; // Vaccine doses increase my resistance to the virus
  // Times patient enters the Latent, Sympt Infected and Recovered states (GE)
  float t_L, t_IS, t_R, t_V1, t_V2;
} Node;

typedef struct List {
  int *v;
  int n;
  int cum[NAGE];
} List;

typedef struct TIMES {
	int id_from;
	int id_to;
	int state_from; 
	int state_to;
} TIMES;

typedef struct Counts {
	int count_l_asymp;
	int count_l_symp;
	int count_l_presymp;
	int count_i_symp;
	int count_recov;
	int count_vacc1;
	int count_vacc2;
  	int countS;
  	int countL;
  	int countIS;
  	int countR;
  	int countV1;
  	int countV2;
	std::vector<int> cvacc1, cvacc2;
	std::vector<int> cS, cL, cIS, cR;
	std::vector<float> times;
} Counts;


typedef struct GSL {
	//Random numbers
	const gsl_rng_type* T;
	gsl_rng *random_gsl;
	gsl_rng *r_rng;  // Why are two needed?
} GSL;

//Parameters
typedef struct Params {
	int N, n_runs, parameters;
	float r, epsilon_asymptomatic, epsilon_symptomatic; 
	float p, gammita, mu, delta, muH, muICU, k, beta_normal;
	float alpha[NAGE], xi[NAGE], beta[NCOMPARTMENTS];
	float dt;
	float vacc1_rate;    //  nb 1st vaccinations per day
	float vacc2_rate;    //  nb 2nd vaccinations per day
	float vacc1_effectiveness; //  % of people for whom 1st shot of the vaccine works as expected
	float vacc2_effectiveness; //  % of people for whom 2nd shot of the vaccine works as expected
	// By default the same as the effectiveness (effect on transmissibility)
	float vacc1_recov_eff; //  reduction in recovery time due to vaccine shot
	float vacc2_recov_eff; //  reduction in recovery time due to vaccine shot
	float dt_btw_vacc; // Time between vacc1 and vacc2
	int max_nb_avail_doses;  // maximum number of available doses
	int nb_doses;      // 1 or 2 depending on the vaccine or experiment peformed
} Params;

//Spreading
typedef struct Lists {
	int n_active, index_node;
	List susceptible; // Not clear whether necessary
	List latent_asymptomatic, latent_symptomatic, infectious_asymptomatic, pre_symptomatic, infectious_symptomatic, home, hospital, icu, recovered, vacc1, vacc2;
	List new_latent_asymptomatic, new_latent_symptomatic, new_infectious_asymptomatic, new_pre_symptomatic, new_infectious_symptomatic, new_home, new_hospital, new_icu, new_recovered, new_vacc1, new_vacc2;
	// Added by GE
	std::vector<int> id_from, id_to, state_from, state_to; //, from_time, to_time;
	std::vector<float> from_time, to_time;
    std::vector<int> people_vaccinated;
	std::vector<int> permuted_nodes;  // randomized population
} Lists;

//Network
typedef struct Network {
	Node *node;
	int start_search; // node to start search from to vaccinate
} Network;

typedef struct Files {
	//Various
	float t;
	int it;  // integer count
	//int t;
	FILE *f_cum, *f_data;
	char f_cum_file[255], f_data_file[255];
	char name_cum[255], name_data[255];
	char data_folder[255], result_folder[255];
	char parameter_file[255];
	char node_file[255];
	char network_file[255];
	char vaccination_file[255];
} Files;

#endif
