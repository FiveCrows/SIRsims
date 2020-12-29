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

typedef struct Node{
  //Info
  int age;
  //Infection
  int state;
  int hospitalization;
  //Connections
  int k;
  int *v;
  double *w;
  // Times patient enters the Latent, Sympt Infected and Recovered states (GE)
  double t_L, t_IS, t_R;
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
  	int countS;
  	int countL;
  	int countIS;
  	int countR;

} Counts;

//States
#define S 0
#define L 1
#define IA 2
#define PS 3
#define IS 4
#define HOME 5
#define H 6
#define ICU 7
#define R 8
// Potentially infected
#define PotL 10

typedef struct GSL {
	//Random numbers
	const gsl_rng_type* T;
	gsl_rng *random_gsl;
	gsl_rng *r_rng;  // Why are two needed?
} GSL;

//Parameters
typedef struct Params {
	int N, n_runs, parameters;
	double r, epsilon_asymptomatic, epsilon_symptomatic, p, gammita, mu, delta, muH, muICU, k, beta_normal;
	double alpha[NAGE], xi[NAGE], beta[NCOMPARTMENTS];
	double dt;
} Params;

//Spreading
typedef struct Lists {
	int n_active, index_node;
	List latent_asymptomatic, latent_symptomatic, infectious_asymptomatic, pre_symptomatic, infectious_symptomatic, home, hospital, icu, recovered;
	List new_latent_asymptomatic, new_latent_symptomatic, new_infectious_asymptomatic, new_pre_symptomatic, new_infectious_symptomatic, new_home, new_hospital, new_icu, new_recovered;
	// Added by GE
	std::vector<int> id_from, id_to, state_from, state_to; //, from_time, to_time;
	std::vector<float> from_time, to_time;
} Lists;

//Network
typedef struct Network {
	Node *node;
} Network;

typedef struct Files {
	//Various
	double t;
	//int t;
	FILE *f_cum, *f_data;
	char f_cum_file[255], f_data_file[255];
	char name_cum[255], name_data[255];
	char data_folder[255], result_folder[255];
	char parameter_file[255];
	char node_file[255];
	char network_file[255];
} Files;

#endif
