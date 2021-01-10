#include <iostream>
#include "stdlib.h"
#include "G.h"
// contains states R,S,L,... that are also defined in earlier
// include files. So must be called last
#include "cxxopts.hpp"
#include "states.h"

#define EXP 1       // Exponential distribution of infection times
#define CONST_INFECTION_TIME 0   // constant recovery time
#define INFECTION_TIME 3.

#define SETUPVACC
#undef SETUPVACC

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
  params.n_runs = 1;
  //parameters = 0;

  seed = time(NULL);
  initRandom(seed, gsl);

  // the Results folder contains parameters.txt, result files
  // the Data folder contains network data
  // executable data_folder result_folder
 
  // reading the parameter file comes before argument parsing. Therefore, 
  // I am hardcoding the name of the parameter file
  strcpy(files.parameter_file, "data_ge/");
  strcat(files.parameter_file, "parameters_0.txt");
  readParameters(files.parameter_file, params);
  parse(argc, argv, params);

#if 0
  if (argc != 3) {
	printf("To run the code, \n");
	printf("  ./executable DataFolder ResultFolder\n");
	printf("-------------------------------------\n");
	exit(1);
  }
#endif

  //strcpy(files.data_folder, argv[1]);
  strcpy(files.data_folder, "data_ge");
  strcat(files.data_folder, "/");
  //strcpy(files.result_folder, argv[2]);
  strcpy(files.result_folder, "data_ge/results/");
  strcat(files.result_folder, "/");

  //strcpy(files.parameter_file, files.result_folder);
  //strcat(files.parameter_file, "parameters_0.txt");
  //printf("parameter_file= %s\n", files.parameter_file); 

  strcpy(files.node_file, files.data_folder);
  strcat(files.node_file, "nodes.txt");
  printf("**** node_file: %s\n", files.node_file);

  strcpy(files.network_file, files.data_folder);
  strcat(files.network_file, "network.txt");
  printf("**** network_file: %s\n", files.network_file);

  strcpy(files.vaccination_file, files.data_folder);
  strcat(files.vaccination_file, "vaccines.csv");

  allocateMemory(params, lists);

  readData(params, lists, network, files);
  setBeta(params);
  printf("end initialize: params.N= %d\n", params.N);
}

//----------------------------------------------------------------------
void G::runSimulation(Params& params, Lists& lists, Counts& c, Network& n, GSL& gsl, Files& f)
{
  for (int run=0; run < params.n_runs; run++) {
     printf("run= %d\n", run);
     {
  		c.count_l_asymp   = 0;
  		c.count_l_symp    = 0;
  		c.count_l_presymp = 0;
  		c.count_i_symp    = 0;
  		c.count_recov     = 0;
  		c.count_vacc1     = 0;
  		c.count_vacc2     = 0;

        init(params, c, n, gsl, lists, f);

        while(lists.n_active>0) {
	      spread(run, f, lists, n, params, gsl, c);
		}

  		printf("total number latent symp:  %d\n", c.count_l_symp);
  		printf("total number latent presymp:  %d\n", c.count_l_presymp);
  		printf("total number infectious symp:  %d\n", c.count_i_symp);
  		printf("total number recovered:  %d\n", c.count_recov);
  		printf("total number vacc 1:  %d\n", c.count_vacc1);
  		printf("total number vacc 2:  %d\n", c.count_vacc2);
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
  c.countV1 = 0;
  c.countV2 = 0;

  for(int i=0;i < params.N; i++) {
    if (n.node[i].state == S)  c.countS++;
    if (n.node[i].state == L)  c.countL++;
    if (n.node[i].state == IS) c.countIS++;
    if (n.node[i].state == R)  c.countR++;
    if (n.node[i].state == V1)  c.countV1++;
    if (n.node[i].state == V2)  c.countV2++;
  }
  printf("Counts: S,L,IS,R: %d, %d, %d, %d\n", c.countS, c.countL, c.countIS, c.countR);
  printf("Counts: V1,V2: %d, %d\n", c.countV1, c.countV2);
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
  int id;
 
  // Use a permutation to make sure that there are no duplicates when 
  // choosing more than one initial infected
  gsl_permutation* p = gsl_permutation_alloc(par.N);
  gsl_rng_env_setup();
  gsl.T = gsl_rng_default;
  gsl.r_rng = gsl_rng_alloc(gsl.T); // *****
  // Set up a random seed
  gsl_rng_set(gsl.r_rng, time(NULL));
  gsl_permutation_init(p);
  // MAKE SURE par.N is correct
  gsl_ran_shuffle(gsl.r_rng, p->data, par.N, sizeof(size_t));


// Initialize Susceptibles. List not needed if there are no vaccinations
  for (int i=0; i < par.N; i++) {
	 // susceptibles have been subject to a random permution 
	 id = p->data[i];  
	 // Nodes in randomized order to randomize batch vaccinations
	 l.permuted_nodes.push_back(p->data[i]);
     n.node[i].state = S;
     addToList(&l.susceptible, id); 
  }

  count_states(par, c, n);
  printf("inside seedInf\n\n\n");

#ifdef SETUPVACC
  int nb_vaccinated = l.people_vaccinated.size();
  int maxN = l.susceptible.n;

  //for (int i=0; i < l.susceptible.n; i++) {
  for (int i=0; i < maxN; i++) {
	id = l.susceptible.v[i];
  	int j = removeFromList(&l.susceptible, id);
  }
#endif

  //for (int i=0; i < 100; i++) {
     //printf("person %d is vaccinated\n");
	 
  // Set up vaccinated people at initial time (optional)
#ifdef SETUPVACC
  // MIGHT HAVE TO FIX THIS
  for (int i=0; i < nb_vaccinated; i++) {
	    int id = l.people_vaccinated[i];
	  	//n.node[id].state = V1;
		n.node[id].t_V1 = 0.;
		// I should remove the correct person from the list of susceptibles
		// Not correct. Only works if loop above is over the same list one
		// is removing from. 
	    //int j = removeFromList(&l.susceptible, i);
        //addToList(&l.vacc1, j); 
  }
  count_states(par, c, n);

  // set Recovered to those vaccinated
  for (int i=0; i < nb_vaccinated; i++) {
	    int person_vaccinated = l.people_vaccinated[i];
	  	n.node[person_vaccinated].state = R;
  }
#endif

  float rho = 0.001; // infectivity percentage at t=0 (SHOULD BE IN PARAMS)
  int ninfected = rho * par.N;
  //ninfected = 1;
  printf("N= %d, ninfected= %d\n", par.N, ninfected);

  for (int i=0; i < ninfected; i++) { 
  	seed = p->data[i];  // randomized infections
  	n.node[seed].state = L;
    addToList(&l.latent_symptomatic, seed); // orig 
	// Not sure. 
	//removeFromList(&l.susceptibles, i);  // i is index to remove
  }
 

  // Count number in recovered state
  int count = 0;
  for (int i=0; i < par.N; i++) {
      if (n.node[i].state == R) { count++; }
  }
  printf("nb recovered: %d\n", count);

  gsl_permutation_free(p);  // free storage

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
  double cur_time = f.t;

  // Vaccinate people at specified daily rate
  // S to V1
  vaccinations(params, l, gsl, net, c, cur_time);

  // Transition from first to second batch
  secondVaccination(par, l, gsl, net, c, cur_time);

  // S to L
  //printf("before infection()\n"); count_states(params, c, net);
  infection(l, net, params, gsl, c, cur_time);
  // L to P
  //printf("before latency()\n"); count_states(params, c, net);
  latency(params, l, gsl, net, c, cur_time);
  // P to I
  //preToI();
  // I to R
  //printf("before IsTransition()\n"); count_states(params, c, net);
  IsTransition(params, l, net, c, gsl, cur_time);


  updateLists(l, net);
  //printf("----------------------------------------\n");
  updateTime();

  //Write data
  fprintf(f.f_data,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %f\n",
	l.susceptible.n, 
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
	l.new_vacc1.n, 
	l.new_vacc2.n, 
	run,
    f.t
	);
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
  //static int count_success = 0;
  //static int count_total   = 0;

  if (net.node[source].state != IS) { // Code did not exit
	printf("I expected source state to be IS instead of %d\n", net.node[source].state); exit(1);
  }

  //printf("infect,source= %d,  ...k= %d\n", source, net.node[source].k);  //all zero
  for (int j=0; j < net.node[source].k; j++) {  
      target = net.node[source].v[j]; // neighbor j of source
	  // Added if branch for tracking potential infected perform more detailed 
	  // measurements of generation time contraction
	  float beta = net.node[source].beta_IS * net.node[source].w[j];;
	  prob = params.dt * beta;
	  //printf("beta= %f, %f\n", beta, net.node[source].beta_IS);
	  //printf("prob= %f\n", prob);

#if EXP
	    prob = 1.-exp(-prob);   // == prob as prob -> zero
#endif
	  if (net.node[target].state != S) {
	      if (gsl_rng_uniform(gsl.random_gsl) < prob) {
		  	stateTransition(source, target, IS, PotL, net.node[source].t_IS, cur_time);
		  }
	  } else {
#if 0
		if (net.node[source].state != IS) {
			printf("I expected source to be IS\n"); exit(1);
			// Code did not exit
		}
		float a = 1. / params.beta_normal;
	    float b = 1. / 0.37;
		float g = gsl_ran_gamma(gsl.r_rng, a, b);  // not yet used
#endif
		//printf("infect: prob= %f\n", prob);
	    if (gsl_rng_uniform(gsl.random_gsl) < prob) {

		    addToList(&l.new_latent_symptomatic, target);
			c.count_l_symp += 1;

		  #if 0
	      if (gsl_rng_uniform(gsl.random_gsl) < params.p) { // p = 0
		    addToList(&l.new_latent_asymptomatic, target);
			c.count_l_asymp += 1;
		  } else {
		    addToList(&l.new_latent_symptomatic, target);
			c.count_l_symp += 1;
  		    count_success++;
			double ratio = (double) count_success / (double) count_total;
			printf("count_success= %d, count_total= %d\n", count_success, count_total);
			printf("ratio: success/total: %f\n", ratio);
		  }
          #endif

		  // Values of time interval distribution are incorrect it seems.
		  // Chances there is an error in the next line
	      // Update target data
	      net.node[target].state = L;
	      net.node[target].t_L   = cur_time;
		  stateTransition(source, target, IS, L, net.node[source].t_IS, net.node[target].t_L);

	      //Various
	      l.n_active++;
	    }
	  } // check whether state is S
  } // for  
}
//----------------------------------------------------------------------
void G::vaccinations(Params& par, Lists& l, GSL& gsl, Network &net, Counts& c, double cur_time)
{
  // SOME KIND OF ERROR. MUST LOOK CAREFULLY AT DEFINITIONS OF RATES
  // Poisson  Pois(lambda), mean(lambda). So lambda is in number/time=rate
	//printf("par.vacc1_rate= %f\n", par.vacc1_rate);
  int n_to_vaccinate = gsl_ran_poisson(gsl.r_rng, par.vacc1_rate*par.dt);
	//printf("Pois, n_to_vaccinate: %d\n", n_to_vaccinate);
  vaccinateNextBatch(net, l, c, par, gsl, n_to_vaccinate, cur_time);
}
//----------------------------------------------------------------------
void G::vaccinateNextBatch(Network& net, Lists& l, Counts& c, Params& par, GSL& gsl, int n, double cur_time) {
// Vaccinate n susceptibles (state == S)

	if (net.start_search == par.N) return;

	int count = 0;
	for (int i=net.start_search; i < par.N; i++) {
		int id = l.permuted_nodes[i];  // This allows vaccinations in randomized order
		if (net.node[id].state == S) {
			net.start_search++;
			count++;
		    c.count_vacc1++;
			//net.node[id].state = V1;
			net.node[id].is_vacc = 1;
			net.node[id].t_V1 = cur_time;
			// constant in time
	        net.node[id].beta_IS = par.beta[IS] * (1.-par.vacc1_effectiveness); 
	        //printf("vacc beta_IS= %f\n", net.node[id].beta_IS);

			// Probability of vaccine effectivness
			// if effectiveness is 1., else branch is always true
	        float prob = par.dt * par.vacc1_effectiveness;
	        if (gsl_rng_uniform(gsl.random_gsl) < prob) {
        		net.node[id].vacc_infect  = 0.;  // transmission rate to others 
        		net.node[id].vacc_suscept = 0.;  // susceptibility to infection
			} else {
        		net.node[id].vacc_infect  = 1.;  // transmission rate to others 
        		net.node[id].vacc_suscept = 1.;  // susceptibility to infection
			}

			//net.node[id].vacc_infect = 
        	//net.node[i].vacc_infect  = 1.0;
			//net.node[i].vacc_suscept = 1.0;
        	//vac1_effect 0.6   # Effective on x% [0,1] of the vaccinated
			// Add to V1 list
			addToList(&l.new_vacc1, id);
		    stateTransition(id, id, S, V1, 0., cur_time); 
		}
		if (count >= n) break;
	}
	if (count < n) {
		printf("Insufficient Susceptibles to Vaccinate\n");
	}
	//printf("nb vaccinated_1: %d\n", l.vacc1.n);
	//printf("start_search= %d\n", net.start_search);
	//printf("n= %d\n", n);
	//printf("count= %d\n", count);

	// Once n is zero, 
	//exit(1);
}
//-------------------------------------------------------------------------------
void G::secondVaccination(Params& par, Lists& l, GSL& gsl, Network &net, Counts& c, double cur_time)
{
	for (int i=0; i < l.vacc1.n; i++) {
		float time_since = cur_time - net.node[l.vacc1.v[i]].t_V1;
		int id = l.vacc1.v[i];
		if (time_since >= par.dt_btw_vacc) {
			addToList(&l.new_vacc2, id);
		    c.count_vacc2 += 1;
	        net.node[id].t_V2 = cur_time;
	        net.node[id].beta_IS = par.beta[IS] * (1.-par.vacc2_effectiveness); 
			i = removeFromList(&l.vacc1, i);
		    stateTransition(id, id, V1, V2, net.node[id].t_V1, cur_time);
		}
	}
    //printf("vacc1.n= %d, vacc2.n= %d\n", l.vacc1.n, l.vacc2.n);
    //printf("new_vacc1.n= %d, new_vacc2.n= %d\n", l.new_vacc1.n, l.new_vacc2.n);
}
//----------------------------------------------------------------------
void G::latency(Params& par, Lists& l, GSL& gsl, Network &net, Counts& c, double cur_time)
{
  int id;

  // prob goes to 1 as eps_S -> 0
#if 1
  for (int i=0; i < l.latent_symptomatic.n; i++) {
      id = l.latent_symptomatic.v[i];  
#if EXP
	  double prob = (1.-exp(-par.dt*par.epsilon_symptomatic));
#else
	  double prob = par.dt*par.epsilon_symptomatic;
#endif
	  //printf("prob L->IS (epsilon_symptomatic): %f, inv: %f\n", prob/par.dt, par.dt/prob);
	  if (gsl_rng_uniform(gsl.random_gsl) < prob) {
	    addToList(&l.new_infectious_symptomatic, id);
		c.count_i_symp += 1;
	    net.node[id].state = IS;
	    net.node[id].t_IS = cur_time;
	    i = removeFromList(&l.latent_symptomatic, i);
		stateTransition(id, id, L, IS, net.node[id].t_L, cur_time);
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
#if CONST_INFECTION_TIME
	if ((cur_time-net.node[id].t_IS) >= INFECTION_TIME) {
#else
 #if EXP
    double prob = 1. - exp(-par.dt*par.mu);
 #else
    double prob = par.dt * par.mu;
 #endif
    if (gsl_rng_uniform(gsl.random_gsl) < prob) { //days to R/Home
#endif
      addToList(&l.new_recovered, id);
	  c.count_recov      += 1;
      net.node[id].state  = R;
	  net.node[id].t_R    = cur_time;
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
  //printf("t = %f\n", files.t);
  //f.t++;
  //printf("     Update Time, t= %d\n", t);
}
//----------------------------------------------------------------------
void G::resetVariables(Lists& l, Files& files)
{  
  l.susceptible.n = 0;
  l.latent_asymptomatic.n = 0;
  l.latent_symptomatic.n = 0;
  l.infectious_asymptomatic.n = 0;
  l.pre_symptomatic.n = 0;
  l.infectious_symptomatic.n = 0;
  l.home.n = 0;
  l.hospital.n = 0;
  l.icu.n = 0;
  l.recovered.n = 0;
  l.vacc1.n = 0;
  l.vacc2.n = 0;

  for(int i=0; i < NAGE; i++)
  {
      l.susceptible.cum[i] = 0;
      l.latent_asymptomatic.cum[i] = 0;
      l.latent_symptomatic.cum[i] = 0;
      l.infectious_asymptomatic.cum[i] = 0;
      l.pre_symptomatic.cum[i] = 0;
      l.infectious_symptomatic.cum[i] = 0;
      l.home.cum[i] = 0;
      l.hospital.cum[i] = 0;
      l.icu.cum[i] = 0;
      l.recovered.cum[i] = 0;
	  l.vacc1.cum[i] = 0;
	  l.vacc2.cum[i] = 0;
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
    for(int i=0; i < par.N; i++) {
        net.node[i].state = S;
		net.node[i].is_vacc = 0;
		net.node[i].beta_IS = par.beta[IS]; 
        net.node[i].vacc_infect  = 1.0;
        net.node[i].vacc_suscept = 1.0;
    }

    net.start_search = 0;
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
  l.new_vacc1.n = 0;
  l.new_vacc2.n = 0;
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
  updateList(&l.vacc1, &l.new_vacc1, n);//VACC1
  updateList(&l.vacc2, &l.new_vacc2, n);//VACC2
}

//----------------------------------------------------------------------
void G::results(int run, Lists& l, Files& f)
{
  //Cumulative values
  for(int i=0;i<NAGE;i++)
    fprintf(f.f_cum,"%d %d %d %d %d %d %d %d %d %d %d %d\n",
       l.latent_asymptomatic.cum[i],
       l.latent_symptomatic.cum[i],
       l.infectious_asymptomatic.cum[i],
       l.pre_symptomatic.cum[i],
       l.infectious_symptomatic.cum[i],
       l.home.cum[i],
       l.hospital.cum[i],
       l.icu.cum[i],
       l.recovered.cum[i],
       l.vacc1.cum[i],
       l.vacc2.cum[i],
       run);
}
//----------------------------------------------------------------------
//G::read
void G::readData(Params& params, Lists& lists, Network& network, Files& files)
{
  // Change params.N according to node file
  printf("readNetwork\n");
  readNetwork(params, lists, network, files);
  printf("readNodes\n");
  readNodes(params, files, network);
  printf("readVaccinations\n");
  // Add parameter to parameter file. Run vaccinations, 0/1
  readVaccinations(params, files, network, lists);
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

  params.N 					  = readInt(f);
  params.r 					  = readFloat(f);
  params.epsilon_asymptomatic = readFloat(f);
  params.epsilon_symptomatic  = readFloat(f); //symptomatic latent period-1
  params.p                    = readFloat(f); //proportion of asymptomatic
  params.gammita              = readFloat(f);
  params.mu                   = readFloat(f);

  for(int i=0; i < NAGE; i++){ //symptomatic case hospitalization ratio
    params.alpha[i] = readFloat(f) ;
  }
  for(int i=0; i < NAGE; i++) {
	params.xi[i]    = readFloat(f);
  }

  params.delta        = readFloat(f);
  params.muH          = readFloat(f);
  params.muICU        = readFloat(f);
  params.k            = readFloat(f);
  params.beta_normal  = readFloat(f);
  params.dt           = readFloat(f);
  params.vacc1_rate   = readFloat(f);
  params.vacc2_rate   = readFloat(f);
  params.vacc1_effectiveness = readFloat(f);
  params.vacc2_effectiveness = readFloat(f);
  params.vacc2_rate   = readFloat(f);
  params.dt_btw_vacc  = readFloat(f);
  printf("read dt_btw_vacc: %f\n", params.dt_btw_vacc);

  params.epsilon_asymptomatic = 1.0/params.epsilon_asymptomatic;
  params.epsilon_symptomatic  = 1.0/params.epsilon_symptomatic;
  params.delta   = 1.0/params.delta;
  params.muH     = 1.0/params.muH;
  params.muICU   = 1.0/params.muICU;
  params.gammita = 1.0/params.gammita;
  params.mu      = 1.0/params.mu;

  for(int i=0; i < NAGE; i++) {
    params.alpha[i] = params.alpha[i]/100;
    params.xi[i]    = params.xi[i]/100;
  }

  fclose(f);
}
//----------------------------------------------------------------------
void G::readNetwork(Params& params, Lists& lists, Network& network, Files& files)
{
  int s, t;
  double w;
  FILE *f;

  FILE* fd = fopen(files.node_file, "r");
  fscanf(fd, "%d", &params.N);
  printf("params.N= %d\n", params.N);

  network.node = (Node*) malloc(params.N * sizeof * network.node);

  for(int i=0;i<params.N;i++)
    {
      network.node[i].k = 0;
      network.node[i].v = (int*) malloc(sizeof * network.node[i].v);
      network.node[i].w = (float*) malloc(sizeof * network.node[i].w);
      network.node[i].t_L  = 0.;
      network.node[i].t_IS = 0.;
      network.node[i].t_R  = 0.;
      network.node[i].t_V1  = -1.;  // uninitialized if negative
      network.node[i].t_V2  = -1.;
    }

  printf("files.network_file= %s\n", files.network_file);
  printf("files.node_file= %s\n", files.node_file);

  f = fopen(files.network_file, "r");
  int nb_edges;
  fscanf(f, "%d", &nb_edges);
  printf("nb_edges= %d\n", nb_edges);

  for (int i=0; i < nb_edges; i++) {
	  fscanf(f, "%d%d%lf", &s, &t, &w);
	  //printf("s,t,w= %d, %d, %f\n", s, t, w);
      network.node[s].k++;
      //Update size of vectors
      network.node[s].v = (int*) realloc(network.node[s].v, network.node[s].k * sizeof *network.node[s].v);
      network.node[s].w = (float*) realloc(network.node[s].w, network.node[s].k * sizeof *network.node[s].w);
      //Write data
      network.node[s].v[network.node[s].k-1] = t;
      network.node[s].w[network.node[s].k-1] = w;

	  // The input data was an undirected graph
	  // This code requires directed edges
	  if (t != s) {
		network.node[t].k++;
      	network.node[t].v = (int*) realloc(network.node[t].v, network.node[t].k * sizeof *network.node[t].v);
        network.node[t].w = (float*) realloc(network.node[t].w, network.node[t].k * sizeof *network.node[t].w);
        network.node[t].v[network.node[t].k-1] = s;
        network.node[t].w[network.node[t].k-1] = w;
	  }
    }
  fclose(f);
}

//----------------------------------------------------------------------
void G::readNodes(Params& params, Files& files, Network& network)
{
  // Ideally, the graph nodes should be randomized before reading them. 
  // Randomization is difficult after read-in
  int age;
  FILE *f;
  int nb_nodes;

  f = fopen(files.node_file, "r");
  fscanf(f, "%d", &nb_nodes);   // WHY is \n required? (only eats up spaces)

  if (params.N != nb_nodes) {
      printf("nb_nodes not equal to params.N. Fix error\n");
	  exit(1);
  }

  int N = 0;
  int node;

  for (int i=0; i < nb_nodes; i++) {
    fscanf(f, "%d%d", &node, &age);
	network.node[node].age = age;
	N++;
  }

  printf("params.N= %d, N= %d (should be identical)\n", params.N, N);
  params.N = N;
  printf("params.N= %d, N= %d (should be identical)\n", params.N, N);

  fclose(f);
}
//----------------------------------------------------------------------
void G::readVaccinations(Params& params, Files& files, Network& network, Lists& l)
{
  FILE *f;
  int nb_nodes;

  f = fopen(files.vaccination_file, "r");
  fscanf(f, "%d", &nb_nodes);

  if (params.N != nb_nodes) {
      printf("nb_nodes (%d) not equal to params.N (%d). Fix error\n", nb_nodes, params.N);
	  exit(1);
  }
 
  int node;
  int* vaccinated = new int [nb_nodes];
  float efficacy;

  for (int i=0; i < nb_nodes; i++) {
	  fscanf(f, "%d%d%f", &node, &vaccinated[i], &efficacy);
	  if (vaccinated[i] == 1) {
		l.people_vaccinated.push_back(i);
	  }
	  //printf("node, vaccinated, efficiency: %d, %d, %f\n", node, vaccinated, efficacy);
  }

  delete [] vaccinated;

  // Set the vaccinated people to recovered

  fclose(f);
}
//----------------------------------------------------------------------
//utilities
void G::allocateMemory(Params& p, Lists& l)
{
  //Spreading
  // Memory: 4.4Mbytes for these lists for Leon County, including vaccines
  printf("memory: %lu bytes\n", p.N * sizeof(*l.latent_asymptomatic.v) * 11 * 2);
  printf("sizeof(int*)= %ld\n", sizeof(int*));
  l.susceptible.v = (int*) malloc(p.N * sizeof * l.susceptible.v);
  l.latent_asymptomatic.v = (int*) malloc(p.N * sizeof * l.latent_asymptomatic.v);
  l.latent_symptomatic.v = (int*) malloc(p.N * sizeof * l.latent_symptomatic.v);
  l.infectious_asymptomatic.v = (int*) malloc(p.N * sizeof * l.infectious_asymptomatic.v);
  l.pre_symptomatic.v = (int*) malloc(p.N * sizeof * l.pre_symptomatic.v);
  l.infectious_symptomatic.v = (int*) malloc(p.N * sizeof * l.infectious_symptomatic.v);
  l.home.v = (int*) malloc(p.N * sizeof * l.home.v);
  l.hospital.v = (int*) malloc(p.N * sizeof * l.hospital.v);
  l.icu.v = (int*) malloc(p.N * sizeof * l.icu.v);
  l.recovered.v = (int*) malloc(p.N * sizeof * l.recovered.v);
  l.vacc1.v = (int*) malloc(p.N * sizeof * l.vacc1.v);
  l.vacc2.v = (int*) malloc(p.N * sizeof * l.vacc2.v);
  
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
  l.new_vacc1.v = (int*) malloc(p.N * sizeof * l.new_vacc1.v);
  l.new_vacc2.v = (int*) malloc(p.N * sizeof * l.new_vacc2.v);
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

  //printf("r= %f\n", p.r);
  //printf("beta_normal= %f\n", p.beta_normal);
  p.beta[IA] = p.r * p.beta_normal;
  p.beta[IS] = p.beta_normal;
  p.beta[PS] = beta_pre;
  p.beta[PS] = 0.0;  // I want infected under these conditions
  //printf("beta[IA] = %f\n", p.beta[IA]);
  //printf("beta[IS] = %f\n", p.beta[IS]);
  //printf("beta[PS] = %f\n", p.beta[PS]);
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

  free(l.susceptible.v);
  free(l.latent_asymptomatic.v);
  free(l.latent_symptomatic.v);
  free(l.infectious_asymptomatic.v);
  free(l.pre_symptomatic.v);
  free(l.infectious_symptomatic.v);
  free(l.home.v);
  free(l.hospital.v);
  free(l.icu.v);
  free(l.recovered.v);
  free(l.vacc1.v);
  free(l.vacc2.v);

  free(l.new_latent_asymptomatic.v);
  free(l.new_latent_symptomatic.v);
  free(l.new_infectious_asymptomatic.v);
  free(l.new_pre_symptomatic.v);
  free(l.new_infectious_symptomatic.v);
  free(l.new_home.v);
  free(l.new_hospital.v);
  free(l.new_icu.v);
  free(l.new_recovered.v);
  free(l.new_vacc1.v);
  free(l.new_vacc2.v);
         
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
//----------------------------------------------------------------------
float G::readFloat(FILE* fd)
{
	char junk[500], junk1[500];
	float f;
	junk[0] = '\0'; 
	junk1[0] = '\0';
	fscanf(fd, "%s%f%[^\n]s", junk, &f, junk1);
	printf("==> readFloat: %s, %f, %s\n", junk, f, junk1);
	return f;
}
//----------------------------------------------------------------------
float G::readInt(FILE* fd)
{
	char junk[500], junk1[500];
	int i;
	junk[0] = '\0'; 
	junk1[0] = '\0';
	fscanf(fd, "%s%d%[^\n]s", junk, &i, junk1);
	printf("%s, %d, %s\n", junk, i, junk1);
	printf("==> readFloat: %s, %d, %s\n", junk, i, junk1);
	return i;
}
//----------------------------------------------------------------------
void G::parse(int argc, char** argv, Params& par)
{
	try {
        cxxopts::Options options(argv[0], " - example command line options");
        options
          .positional_help("[optional args]")
          .show_positional_help();
    
        options
          //.allow_unrecognised_options()
          .add_options()
		  ("N", "Number of nodes", cxxopts::value<int>())
		  ("gamma", "Average recovery time", cxxopts::value<float>())
		  ("mu", " Average latent time", cxxopts::value<float>())
		  ("betaIS", " Transmissibility rate", cxxopts::value<float>())
		  ("dt", " Time step", cxxopts::value<float>())
		  ("vac1_rate", " Rate of 1st vaccine dose", cxxopts::value<int>())
		  ("vac2_rate", " Rate of 2nd vaccine dose", cxxopts::value<int>())
		  ("vac1_eff", " First vaccine dose efficacy [0-1]", cxxopts::value<float>())
		  ("vac2_eff", " Second vaccine dose efficacy [0-1]", cxxopts::value<float>())
		  ("dt_btw_vacc", " Time between 1st and 2nd vaccine doses", cxxopts::value<float>())
          ("help", "Print help")
        ;

    	auto res = options.parse(argc, argv);

		if (res.count("N") == 1)
			par.N = res["N"].as<int>();
		if (res.count("gamma") == 1)
			par.gammita = res["gamma"].as<float>();
		if (res.count("mu") == 1)
			par.mu = res["mu"].as<float>();
		if (res.count("betaIS") == 1)
			par.beta_normal = res["betaIS"].as<float>();
		if (res.count("dt") == 1)
			par.dt = res["dt"].as<float>();
		if (res.count("vac1_rate") == 1)
			par.vacc1_rate = res["vac1_rate"].as<int>();
			printf("===> par.vacc1_rate= %f\n", par.vacc1_rate);
		if (res.count("vac2_rate") == 1)
			par.vacc2_rate = res["vac2_rate"].as<int>();
		if (res.count("vac1_eff") == 1)
			par.vacc1_effectiveness = res["vac1_eff"].as<float>();
		if (res.count("vac2_eff") == 1)
			par.vacc2_effectiveness = res["vac2_eff"].as<float>();
		if (res.count("dt_btw_vacc") == 1)
			par.dt_btw_vacc = res["dt_btw_vacc"].as<float>();

    //auto arguments = res.arguments();
    } catch(const cxxopts::OptionException& e) {
    		printf("error parse on options\n");
    		std::cout << "error parsing options: " << e.what() << std::endl;
    		exit(1);
	}
	
	return;
}

