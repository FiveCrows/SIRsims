#include "stdlib.h"
#include "G.h"
#include "head.h"

#define EXP 0

using namespace std;

G::G(Params& p, Files& f, Lists& l, Counts& c, Network& n, GSL& gsl) : par(p), files(f), lists(l), net(n), counts(c), gsl(gsl)
{
}

//----------------------------------------------------------------------
void G::initialize(int argc, char** argv, Files& files, Params& params, Lists& lists, Network& network, GSL& gsl)
{
  int seed;

  params.n_runs = 100;
  params.n_runs = 10;
  params.n_runs = 5;
  params.n_runs = 2;
  //parameters = 0;

  seed = time(NULL);
  initRandom(seed, gsl);

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
void G::runSimulation(Params& params, Lists& lists, Counts& c, Network& n, GSL& gsl, Files& f)
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

        init(params, c, n, gsl, lists, f);

        while(lists.n_active>0) {
	      spread(run, f, lists, n, params, gsl, c);
		}

  		printf("total number latent symp:  %d\n", count.count_l_symp);
  		printf("total number latent presymp:  %d\n", count.count_l_presymp);
  		printf("total number infectious symp:  %d\n", count.count_i_symp);
  		printf("total number recovered:  %d\n", count.count_recov);
        results(run, lists, f);

		printTransitionStats();
     }
  }
}
//----------------------------------------------------------------------
//spread
void G::count_states(Params& params, Counts& c, Network& n)
{
  c.countS  = 0;
  c.countL  = 0;
  c.countIS = 0;
  c.countR  = 0;

  for(int i=0;i < params.N; i++) {
    if (n.node[i].state == S)  c.countS++;
    if (n.node[i].state == L)  c.countL++;
    if (n.node[i].state == IS) c.countIS++;
    if (n.node[i].state == R)  c.countR++;
  }
  printf("Counts: S,L,IS,R: %d, %d, %d, %d\n", c.countS, c.countL, c.countIS, c.countR);
}
//----------------------------------------------------------------------
void G::init(Params& p, Counts& c, Network& n, GSL& gsl, Lists& l, Files& f)
{
  resetVariables(l, f);
  resetNodes(p, n);
  printf("after resetN\n");
  
  //Start
  seedInfection(p, c, n, gsl, l, f);
  printf("after seed\n");
}
//----------------------------------------------------------------------
void G::seedInfection(Params& par, Counts& c, Network& n, GSL& gsl, Lists& l, Files& f)
{
  int seed;
  count_states(par, c, n);
  printf("inside seedInf\n");

  // Use a permutation to make sure that there are no duplicates when 
  // choosing more than one initial infected
  gsl_permutation* p = gsl_permutation_alloc(par.N);
  gsl_rng_env_setup();
  gsl.T = gsl_rng_default;
  gsl.r_rng = gsl_rng_alloc(gsl.T); // *****
  gsl_permutation_init(p);
  gsl_ran_shuffle(gsl.r_rng, p->data, par.N, sizeof(size_t));

  float rho = 0.001; // infectivity percentage at t=0 (SHOULD BE IN PARAMS)
  int ninfected = rho * par.N;
  printf("ninfected= %d\n", ninfected);

  for (int i=0; i < ninfected; i++) { 
  	seed = p->data[i];
  	n.node[seed].state = L;
    addToList(&l.latent_symptomatic, seed); // orig 
    //stateTransition(seed, seed, S, L, -1, 0);
  }

  gsl_permutation_free(p);

  l.n_active = ninfected;
  c.count_l_symp += ninfected;
  printf("added %d latent_symptomatic individuals\n", ninfected);

  f.t = par.dt;
}
//----------------------------------------------------------------------
//void G::spread(int){}
void G::spread(int run, Files& f, Lists& l, Network& net, Params& params, GSL& gsl, Counts& c)
{  
  resetNew(l);
  //int cur_time = f.t;
  double cur_time = f.t;

  // S to L
  //printf("before infection()\n"); count_states();
  infection(l, net, params, gsl, c, cur_time);
  // L to P
  //printf("before latency()\n"); count_states();
  latency(params, l, gsl, net, c, cur_time);
  // P to I
  //preToI();
  // I to R
  //printf("before IsTransition()\n"); count_states();
  IsTransition(params, l, net, c, gsl, cur_time);


  updateLists(l, net);
  printf("----------------------------------------\n");
  updateTime();

  //Write data
  fprintf(f.f_data,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
	l.latent_asymptomatic.n, 
	l.latent_symptomatic.n, 
	l.infectious_asymptomatic.n, 
	l.pre_symptomatic.n, 
	l.infectious_symptomatic.n, 
	l.home.n, 
	l.hospital.n, 
	l.icu.n, 
	l.recovered.n, 
	l.new_latent_asymptomatic.n, 
	l.new_latent_symptomatic.n, 
	l.new_infectious_asymptomatic.n, 
	l.new_pre_symptomatic.n, 
	l.new_infectious_symptomatic.n, 
	l.new_home.n, 
	l.new_hospital.n, 
	l.new_icu.n, 
	l.new_recovered.n, 
	run);
}
//----------------------------------------------------------------------
void G::infection(Lists& l, Network& net, Params& params, GSL& gsl, Counts& c, double cur_time)
{
  if (l.infectious_asymptomatic.n > 0) {printf("infectious_asymptomatic should be == 0\n"); exit(1); }
  if (l.pre_symptomatic.n > 0) {printf("pre_symptomatic should be == 0\n"); exit(1); }

  //Infectious symptomatic (check the neighbors of all infected)
  for (int i=0; i < l.infectious_symptomatic.n; i++) {
	//printf("call infect(infectious_symptomatic) %d\n", i);
    infect(l.infectious_symptomatic.v[i], IS, net, params, gsl, l, c, cur_time);
  }
}
//----------------------------------------------------------------------
void G::infect(int source, int type, Network& net, Params& params, GSL& gsl, Lists& l, Counts& c, double cur_time)
{
  int target;
  double prob;
  //printf("p= %f\n", p);
 
  for (int j=0; j < net.node[source].k; j++) { // for
      target = net.node[source].v[j];
	  //printf("j= %d\n", j);
      if (net.node[target].state == S) {   // == S
		// There is an implicit dt factor (== 1 day)
	    //printf("  An infected found S\n");

		float a = 1. / params.beta_normal;
		//float a = 2.23;
	    float b = 1. / 0.37;
		float g = gsl_ran_gamma(gsl.r_rng, a, b);
#if EXP
	    prob = 1.-exp(-params.dt*params.beta[type] * net.node[source].w[j]);
#else
	    prob = params.dt * params.beta[type] * net.node[source].w[j];
#endif
		printf("1 params.dt= %f\n", params.dt);
	  
	    if (gsl_rng_uniform(gsl.random_gsl) < prob) {
	      //Check if asymptomatic

	      if (gsl_rng_uniform(gsl.random_gsl) < params.p) {
		    addToList(&l.new_latent_asymptomatic, target);

			c.count_l_asymp += 1;
		  } else {
		    addToList(&l.new_latent_symptomatic, target);
			c.count_l_symp += 1;
		  }

		  stateTransition(source, target, IS, L, net.node[source].t_IS, cur_time);
		  // People are susceptible from time 0.
		  //stateTransition(target, target,  S, L, 0, cur_time);

	      //Update target data
	      net.node[target].state = L;
	      net.node[target].t_L = cur_time;

	      //Various
	      l.n_active++;
	    }
	  } // == S
  } // for  
}
//----------------------------------------------------------------------
void G::latency(Params& par, Lists& l, GSL& gsl, Network &net, Counts& c, double cur_time)
{
  int id;

  // prob goes to 1 as eps_S -> 0
  //printf("prob: %f, n= %d\n", 1.-exp(-epsilon_symptomatic), latent_symptomatic.n);
#if 1
  for (int i=0; i < l.latent_symptomatic.n; i++) {
      id = l.latent_symptomatic.v[i];  
	printf("2 params.dt= %f\n", par.dt);
#if EXP
	  if (gsl_rng_uniform(gsl.random_gsl) < (1.-exp(-par.dt*par.epsilon_symptomatic)))
#else
	  if (gsl_rng_uniform(gsl.random_gsl) < par.dt*par.epsilon_symptomatic)
#endif
	  {
	    addToList(&l.new_infectious_symptomatic, id);
		c.count_i_symp += 1;
	    net.node[id].state = IS;
	    net.node[id].t_IS = cur_time;
	    i = removeFromList(&l.latent_symptomatic, i);

		stateTransition(id, id, L, IS, net.node[id].t_L, cur_time);
	  } else {
		 ;
	  }
  }
#endif
}
//----------------------------------------------------------------------
void G::IaToR(){}
//----------------------------------------------------------------------
void G::preToI(){}
//----------------------------------------------------------------------
void G::IsTransition(Params& par, Lists& l, Network& net, Counts& c, GSL& gsl, double cur_time)
{
  int id;

  // Modified by GE to implement simple SIR model. No hospitalizations, ICU, etc.
  // Go from IS to R
  for (int i=0; i < l.infectious_symptomatic.n; i++) {
    id = l.infectious_symptomatic.v[i];  
#if EXP
    double prob = 1. - exp(-par.dt*par.mu);
#else
    double prob = par.dt * par.mu;
#endif
    if (gsl_rng_uniform(gsl.random_gsl) < prob) { //days to R/Home
      addToList(&l.new_recovered, id);
	  c.count_recov += 1;;
      net.node[id].state = R;
	  net.node[id].t_R = cur_time;
      l.n_active--;
      i = removeFromList(&l.infectious_symptomatic, i);

	  stateTransition(id, id, IS, R, net.node[id].t_IS, cur_time);
	}
  }
  return;
}
//----------------------------------------------------------------------
void G::homeTransition(){}
//----------------------------------------------------------------------
void G::hospitals(){}
//----------------------------------------------------------------------
void G::updateTime()
{ 
  files.t += par.dt;
  printf("t = %f\n", files.t);
  //f.t++;
  //printf("     Update Time, t= %d\n", t);
}
//----------------------------------------------------------------------
void G::resetVariables(Lists& l, Files& files)
{  
  l.latent_asymptomatic.n = 0;
  l.latent_symptomatic.n = 0;
  l.infectious_asymptomatic.n = 0;
  l.pre_symptomatic.n = 0;
  l.infectious_symptomatic.n = 0;
  l.home.n = 0;
  l.hospital.n = 0;
  l.icu.n = 0;
  l.recovered.n = 0;

  for(int i=0;i<NAGE;i++)
    {
      l.latent_asymptomatic.cum[i] = 0;
      l.latent_symptomatic.cum[i] = 0;
      l.infectious_asymptomatic.cum[i] = 0;
      l.pre_symptomatic.cum[i] = 0;
      l.infectious_symptomatic.cum[i] = 0;
      l.home.cum[i] = 0;
      l.hospital.cum[i] = 0;
      l.icu.cum[i] = 0;
      l.recovered.cum[i] = 0;
    }

  files.t = 0;
  l.n_active = 0;

  l.id_from.resize(0);
  l.id_to.resize(0);
  l.state_from.resize(0);
  l.state_to.resize(0);
  l.from_time.resize(0);
  l.to_time.resize(0);
}
//----------------------------------------------------------------------
void G::resetNodes(Params& par, Network& net)
{
  for(int i=0; i < par.N; i++)
    net.node[i].state = S;
}
//----------------------------------------------------------------------
void G::resetNew(Lists& l)
{
  l.new_latent_asymptomatic.n = 0;
  l.new_latent_symptomatic.n = 0;
  l.new_infectious_asymptomatic.n = 0;
  l.new_pre_symptomatic.n = 0;
  l.new_infectious_symptomatic.n = 0;
  l.new_home.n = 0;
  l.new_hospital.n = 0;
  l.new_icu.n = 0;
  l.new_recovered.n = 0;
}
//----------------------------------------------------------------------
void G::updateLists(Lists& l, Network& n)
{
  updateList(&l.latent_asymptomatic, &l.new_latent_asymptomatic, n);//LA
  updateList(&l.latent_symptomatic, &l.new_latent_symptomatic, n);//LS
  updateList(&l.infectious_asymptomatic, &l.new_infectious_asymptomatic, n);//IA
  updateList(&l.pre_symptomatic, &l.new_pre_symptomatic, n);//PS
  updateList(&l.infectious_symptomatic, &l.new_infectious_symptomatic, n);//IS
  updateList(&l.home, &l.new_home, n); //Home
  updateList(&l.hospital, &l.new_hospital, n);//H
  updateList(&l.icu, &l.new_icu, n);//ICU
  updateList(&l.recovered, &l.new_recovered, n);//R
}

//----------------------------------------------------------------------
void G::results(int run, Lists& l, Files& f)
{
  //Cumulative values
  for(int i=0;i<NAGE;i++)
    fprintf(f.f_cum,"%d %d %d %d %d %d %d %d %d %d \n",
       l.latent_asymptomatic.cum[i],
       l.latent_symptomatic.cum[i],
       l.infectious_asymptomatic.cum[i],
       l.pre_symptomatic.cum[i],
       l.infectious_symptomatic.cum[i],
       l.home.cum[i],
       l.hospital.cum[i],
       l.icu.cum[i],
       l.recovered.cum[i],
       run);
}
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
  fscanf(f,"%s %lf", trash, &params.dt); //discrete time step
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
      network.node[i].t_L  = -1;
      network.node[i].t_IS = -1;
      network.node[i].t_R  = -1;
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
void G::initRandom(int seed, GSL& gsl)
{
  //GSL gsl;
  gsl_rng_env_setup();
  gsl.T = gsl_rng_default;
  gsl.random_gsl = gsl_rng_alloc(gsl.T);
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
void G::printTransitionStats()
{
	Lists& l = lists;

	  int lg = l.id_from.size();
	  FILE* fd = fopen("transition_stats.csv", "w");
	  fprintf(fd, "from_id,to_id,from_state,to_state,from_time,to_time\n");
	  for (int i=0; i < lg; i++) {
		 fprintf(fd, "%d, %d, %d, %d, %f, %f\n", l.id_from[i], l.id_to[i], l.state_from[i], l.state_to[i], l.from_time[i], l.to_time[i]);
	  }
	  fclose(fd);
}
//----------------------------------------------------------------------
void G::stateTransition(int source, int target, int from_state, int to_state, double from_time, double to_time)
{
	Lists& l = lists; 
	l.id_from.push_back(source);
	l.id_to.push_back(target);
	l.state_from.push_back(from_state);
	l.state_to.push_back(to_state);
	l.from_time.push_back(from_time);
	l.to_time.push_back(to_time);
}
