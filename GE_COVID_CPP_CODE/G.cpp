#include "stdlib.h"
#include "G.h"
#include "head.h"

using namespace std;

//----------------------------------------------------------------------
void G::initialize(int argc, char** argv, Files& files, Params& params, Lists& lists, Network& network)
{
  int seed;

  params.n_runs = 100;
  params.n_runs = 10;
  params.n_runs = 1;
  params.n_runs = 5;
  //parameters = 0;

  seed = time(NULL);
  initRandom(seed);

  // the Results folder contains parameters.txt, result files
  // the Data folder contains network data
  // executable data_folder result_folder

  if (argc != 3) {
	printf("To run the code, \n");
	printf("  ./executable DataFolder ResultFolder\n");
	printf("-------------------------------------\n");
	exit(1);
  }

  strcpy(files.data_folder, argv[1]);
  strcat(files.data_folder, "/");
  strcpy(files.result_folder, argv[2]);
  strcat(files.result_folder, "/");
  //printf("result_folder: %s\n", result_folder); 
  //printf("data_folder: %s\n", data_folder); 

  //char* filenm = "parameters_0.txt";
  strcpy(files.parameter_file, files.result_folder);
  strcat(files.parameter_file, "parameters_0.txt");
  printf("parameter_file= %s\n", files.parameter_file); 

  strcpy(files.node_file, files.data_folder);
  strcat(files.node_file, "nodes.txt");

  strcpy(files.network_file, files.data_folder);
  strcat(files.network_file, "network.txt");

  readParameters(files.parameter_file, params);
  allocateMemory(params, lists);
  readData(params, lists, network, files);
  setBeta(params);
}

//----------------------------------------------------------------------
void G::runSimulation(Params& params, Lists& lists)
{
  Counts count;

  for (int run=0; run < params.n_runs; run++) {
     printf("run= %d\n", run);
     {
  		count.count_l_asymp   = 0;
  		count.count_l_symp    = 0;
  		count.count_l_presymp = 0;
  		count.count_i_symp    = 0;
  		count.count_recov     = 0;

        init();

        while(lists.n_active>0) {
          //printf("runSinm n_active= %d\n", n_active);
	      spread(run);
		}

  		printf("total number latent symp:  %d\n", count.count_l_symp);
  		printf("total number latent presymp:  %d\n", count.count_l_presymp);
  		printf("total number infectious symp:  %d\n", count.count_i_symp);
  		printf("total number recovered:  %d\n", count.count_recov);
        results(run);
     }
  }
}
//----------------------------------------------------------------------
//spread
void G::init(){}
//----------------------------------------------------------------------
void G::seedInfection(){}
//----------------------------------------------------------------------
void G::spread(int){}
//----------------------------------------------------------------------
void G::infection(){}
//----------------------------------------------------------------------
void G::infect(int, int){}
//----------------------------------------------------------------------
void G::latency(){}
//----------------------------------------------------------------------
void G::IaToR(){}
//----------------------------------------------------------------------
void G::preToI(){}
//----------------------------------------------------------------------
void G::IsTransition(){}
//----------------------------------------------------------------------
void G::homeTransition(){}
//----------------------------------------------------------------------
void G::hospitals(){}
//----------------------------------------------------------------------
void G::updateTime(){}
//----------------------------------------------------------------------
void G::resetVariables(){}
//----------------------------------------------------------------------
void G::resetNodes(){}
//----------------------------------------------------------------------
void G::resetNew(){}
//----------------------------------------------------------------------
void G::updateLists(){}
//----------------------------------------------------------------------
void G::results(int){}
//----------------------------------------------------------------------
//G::read
void G::readData(Params& params, Lists& lists, Network& network, Files& files)
{
  readNetwork(params, lists, network, files);
  readNodes(params, files, network);
}
//----------------------------------------------------------------------
void G::readParameters(char* parameter_file, Params& params)
{
  FILE *f;
  char trash[100];

  int parameters = 0; // fake entry
  sprintf(trash, parameter_file, parameters);
  printf("parameter_file= %s\n", parameter_file);
  f = fopen(trash, "r");
  fscanf(f,"%s %d", trash, &params.N); //Nodes
  fscanf(f,"%s %lf", trash, &params.r); //relative inf. of asymptomatic individuals
  fscanf(f,"%s %lf", trash, &params.epsilon_asymptomatic); //asymptomatic latent period-1
  params.epsilon_asymptomatic = 1.0/params.epsilon_asymptomatic;
  fscanf(f,"%s %lf", trash, &params.epsilon_symptomatic); //symptomatic latent period-1
  params.epsilon_symptomatic = 1.0/params.epsilon_symptomatic;
  fscanf(f,"%s %lf", trash, &params.p); //proportion of asymptomatic
  fscanf(f,"%s %lf", trash, &params.gammita); //pre-symptomatic period
  params.gammita = 1.0/params.gammita;
  fscanf(f,"%s %lf", trash, &params.mu); //time to recover
  params.mu = 1.0/params.mu;
  for(int i=0;i<NAGE;i++){ //symptomatic case hospitalization ratio
    fscanf(f,"%s %lf", trash, params.alpha+i);
    params.alpha[i] = params.alpha[i]/100;
  }
  for(int i=0;i<NAGE;i++){
    fscanf(f,"%s %lf", trash, params.xi+i); //ICU ratio
    params.xi[i] = params.xi[i]/100;
  }
  fscanf(f,"%s %lf", trash, &params.delta); //time to hospital
  params.delta = 1.0/params.delta;
  fscanf(f,"%s %lf", trash, &params.muH); //recovery in hospital
  params.muH = 1.0/params.muH;
  fscanf(f,"%s %lf", trash, &params.muICU); //recovery in ICU
  params.muICU = 1.0/params.muICU;
  fscanf(f,"%s %lf", trash, &params.k); //k
  fscanf(f,"%s %lf", trash, &params.beta_normal); //infectivity
  printf("fscanf, beta_normal= %lf\n", params.beta_normal);
  printf("fscanf, r= %lf\n", params.r);
  fclose(f);
}
//----------------------------------------------------------------------
void G::readNetwork(Params& params, Lists& lists, Network& network, Files& files)
{
  int s, t;
  double w;
  FILE *f;
  char *token;
  char string[500];
  
  network.node = (Node*) malloc(params.N * sizeof * network.node);
  //node = new [N] Node; 
  for(int i=0;i<params.N;i++)
    {
      network.node[i].k = 0;
      network.node[i].v = (int*) malloc(sizeof * network.node[i].v);
      network.node[i].w = (double*) malloc(sizeof * network.node[i].w);
    }

  //f = fopen("Data/network.txt","r");
  //f = fopen("Data_SIR/edges_BA.csv", "r");
  printf("network_file= %s\n", files.network_file);
  f = fopen(files.network_file, "r");
  while(fgets(string,500,f))
    {
      token = strtok(string," ");
      s = atoi(token);
      token = strtok(NULL," ");
      t = atoi(token);
      token = strtok(NULL,"\n");
      w = atof(token);

      network.node[s].k++;
      //Update size of vectors
      network.node[s].v = (int*) realloc(network.node[s].v, network.node[s].k * sizeof *network.node[s].v);
      network.node[s].w = (double*) realloc(network.node[s].w, network.node[s].k * sizeof *network.node[s].w);
      //Write data
      network.node[s].v[network.node[s].k-1] = t;
      network.node[s].w[network.node[s].k-1] = w;

	  // The input data was an undirected graph
	  if (t != s) {
		network.node[t].k++;
      	network.node[t].v = (int*) realloc(network.node[t].v, network.node[t].k * sizeof *network.node[t].v);
        network.node[t].w = (double*) realloc(network.node[t].w, network.node[t].k * sizeof *network.node[t].w);
        network.node[t].v[network.node[t].k-1] = s;
        network.node[t].w[network.node[t].k-1] = w;
	  }
    }
  printf("exit while\n");
  fclose(f);
}

//----------------------------------------------------------------------
void G::readNodes(Params& params, Files& files, Network& network)
{
  int s, age;
  FILE *f;
  char *token;
  char string[500];
  
  //f = fopen("Data/nodes.txt", "r");
  f = fopen(files.node_file, "r");
  while(fgets(string,500,f))
    {
      token = strtok(string," ");
      s = atoi(token);
      token = strtok(NULL,"\n");
      age = atoi(token);

      network.node[s].age = age;
    }
  fclose(f);
}
//----------------------------------------------------------------------
//utilities
void G::allocateMemory(Params& p, Lists& l)
{
  //Spreading
  l.latent_asymptomatic.v = (int*) malloc(p.N * sizeof * l.latent_asymptomatic.v);
  l.latent_symptomatic.v = (int*) malloc(p.N * sizeof * l.latent_symptomatic.v);
  l.infectious_asymptomatic.v = (int*) malloc(p.N * sizeof * l.infectious_asymptomatic.v);
  l.pre_symptomatic.v = (int*) malloc(p.N * sizeof * l.pre_symptomatic.v);
  l.infectious_symptomatic.v = (int*) malloc(p.N * sizeof * l.infectious_symptomatic.v);
  l.home.v = (int*) malloc(p.N * sizeof * l.home.v);
  l.hospital.v = (int*) malloc(p.N * sizeof * l.hospital.v);
  l.icu.v = (int*) malloc(p.N * sizeof * l.icu.v);
  l.recovered.v = (int*) malloc(p.N * sizeof * l.recovered.v);
  
  //New spreading
  l.new_latent_asymptomatic.v = (int*) malloc(p.N * sizeof * l.new_latent_asymptomatic.v);
  l.new_latent_symptomatic.v = (int*) malloc(p.N * sizeof * l.new_latent_symptomatic.v);
  l.new_infectious_asymptomatic.v = (int*) malloc(p.N * sizeof * l.new_infectious_asymptomatic.v);
  l.new_pre_symptomatic.v = (int*) malloc(p.N * sizeof * l.new_pre_symptomatic.v);
  l.new_infectious_symptomatic.v = (int*) malloc(p.N * sizeof * l.new_infectious_symptomatic.v);
  l.new_home.v = (int*) malloc(p.N * sizeof * l.new_home.v);
  l.new_hospital.v = (int*) malloc(p.N * sizeof * l.new_hospital.v);
  l.new_icu.v = (int*) malloc(p.N * sizeof * l.new_icu.v);
  l.new_recovered.v = (int*) malloc(p.N * sizeof * l.new_recovered.v);
}
//----------------------------------------------------------------------
void G::initRandom(int seed)
{
  GSL gsl;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  gsl.random_gsl = gsl_rng_alloc(T);
  gsl_rng_set(gsl.random_gsl,seed);
}

//----------------------------------------------------------------------
//void G::openFiles(char* cum_baseline, char* data_baseline)
void G::openFiles(Files& files)
{
  //char name_cum[100], name_data[100];

  int parameters = 0;  // Do not know where this comes from
  sprintf(files.name_cum, "%s/cum_baseline_p%d.txt", files.result_folder, parameters);
  sprintf(files.name_data, "%s/data_baseline_p%d.txt",files.result_folder, parameters);

  //sprintf(name_cum,"Results/cum_baseline_p%d.txt",parameters);
  //sprintf(name_data,"Results/data_baseline_p%d.txt",parameters);

  files.f_cum = fopen(files.name_cum,"w");
  files.f_data = fopen(files.name_data,"w");
}
//----------------------------------------------------------------------
void G::setBeta(Params& p)
{
  double beta_pre;

  beta_pre = p.beta_normal * p.gammita*p.k/(p.mu*(1-p.k));
    
  for(int i=0; i < NCOMPARTMENTS; i++)
    p.beta[i] = 0;

  printf("r= %f\n", p.r);
  printf("beta_normal= %f\n", p.beta_normal);
  p.beta[IA] = p.r * p.beta_normal;
  p.beta[IS] = p.beta_normal;
  p.beta[PS] = beta_pre;
  p.beta[PS] = 0.0;  // I want infected under these conditions
  printf("beta[IA] = %f\n", p.beta[IA]);
  printf("beta[IS] = %f\n", p.beta[IS]);
  printf("beta[PS] = %f\n", p.beta[PS]);
}

//----------------------------------------------------------------------
void G::addToList(List *list, int id)
{
  list->v[list->n] = id;
  list->n++;
}
//----------------------------------------------------------------------
int G::removeFromList(List *list, int i)
{
  list->n--;
  list->v[i] = list->v[list->n];

  return i-1;
}

//----------------------------------------------------------------------
void G::updateList(List* list, List *newl, Network& network)
{
  for(int i=0;i<newl->n;i++)
    {
      list->v[list->n] = newl->v[i];
      list->n++;
      list->cum[network.node[newl->v[i]].age]++;
    }
}
//----------------------------------------------------------------------
void G::closeFiles(Files& files)
{
  fclose(files.f_cum);
  fclose(files.f_data);
}
//----------------------------------------------------------------------
void G::print(int t0)
{
  printf("Execution time %d seconds\n",(int)time(NULL)-t0);
}
//----------------------------------------------------------------------
void G::freeMemory(Params& p, Network& network, Lists& l, GSL& gsl)
{
  for(int i=0;i< p.N;i++)
    {
      free(network.node[i].v);
      free(network.node[i].w);
    }
  free(network.node);

  free(l.latent_asymptomatic.v);
  free(l.latent_symptomatic.v);
  free(l.infectious_asymptomatic.v);
  free(l.pre_symptomatic.v);
  free(l.infectious_symptomatic.v);
  free(l.home.v);
  free(l.hospital.v);
  free(l.icu.v);
  free(l.recovered.v);

  free(l.new_latent_asymptomatic.v);
  free(l.new_latent_symptomatic.v);
  free(l.new_infectious_asymptomatic.v);
  free(l.new_pre_symptomatic.v);
  free(l.new_infectious_symptomatic.v);
  free(l.new_home.v);
  free(l.new_hospital.v);
  free(l.new_icu.v);
  free(l.new_recovered.v);
         
  gsl_rng_free(gsl.random_gsl);
}
//----------------------------------------------------------------------
