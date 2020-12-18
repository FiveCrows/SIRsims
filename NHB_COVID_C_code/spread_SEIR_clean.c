#include "head.h"

#define EXP 1

// Implement an SEIR version of this code. 
// There are 4 states: Susceptible, Exposed (latent), Infected, Recovered
// Recovered means death or cured. 

void count_states()
{
  int countS  = 0;
  int countL  = 0;
  int countIS = 0;
  int countR  = 0;

  for(int i=0;i<N;i++) {
    if (node[i].state == S)  countS++;
    if (node[i].state == L)  countL++;
    if (node[i].state == IS) countIS++;
    if (node[i].state == R)  countR++;
  }
  printf("Counts: S,L,IS,R: %d, %d, %d, %d\n", countS, countL, countIS, countR);
}

void init()
{
  resetVariables();
  resetNodes();
  
  //Start
  seedInfection();
}

void seedInfection()
{
  int seed;
  count_states();

  // Use a permutation to make sure that there are no duplicates when 
  // choosing more than one initial infected
  gsl_permutation* p = gsl_permutation_alloc(N);
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r_rng = gsl_rng_alloc(T);
  gsl_permutation_init(p);
  gsl_ran_shuffle(r_rng, p->data, N, sizeof(size_t));

  float rho = 0.001; // infectivity percentage at t=0
  int ninfected = rho * N;
  printf("ninfected= %d\n", ninfected);

  for (int i=0; i < ninfected; i++) { 
  	seed = p->data[i];
  	node[seed].state = L;
	printf("%d, seed= %d\n", i, seed); 
    addToList(&latent_symptomatic, seed); // orig 
  }

  gsl_permutation_free(p);

  n_active = ninfected;
  count_l_symp += ninfected;
  printf("added %d latent_symptomatic individuals\n", ninfected);

  t = 1;
}

void spread(int run)
{  
  resetNew();

  // S to L
  //printf("before infection()\n"); count_states();
  infection();
  // L to P
  //printf("before latency()\n"); count_states();
  latency();
  // P to I
  //preToI();
  // I to R
  //printf("before IsTransition()\n"); count_states();
  IsTransition();


  updateLists();
  printf("----------------------------------------\n");
  updateTime();

  //Write data
  fprintf(f_data,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
	latent_asymptomatic.n, 
	latent_symptomatic.n, 
	infectious_asymptomatic.n, 
	pre_symptomatic.n, 
	infectious_symptomatic.n, 
	home.n, 
	hospital.n, 
	icu.n, 
	recovered.n, 
	new_latent_asymptomatic.n, 
	new_latent_symptomatic.n, 
	new_infectious_asymptomatic.n, 
	new_pre_symptomatic.n, 
	new_infectious_symptomatic.n, 
	new_home.n, 
	new_hospital.n, 
	new_icu.n, 
	new_recovered.n, 
	run);
}

void infection()
{
  if (infectious_asymptomatic.n > 0) {printf("infectious_asymptomatic should be == 0\n"); exit(1); }
  if (pre_symptomatic.n > 0) {printf("pre_symptomatic should be == 0\n"); exit(1); }

  //Infectious symptomatic (check the neighbors of all infected)
  for (int i=0; i < infectious_symptomatic.n; i++) {
	//printf("call infect(infectious_symptomatic) %d\n", i);
    infect(infectious_symptomatic.v[i], IS);
  }
}


void infect(int source, int type)
{
  int target;
  double prob;
  //printf("p= %f\n", p);
 
  for (int j=0; j < node[source].k; j++) { // for
      target = node[source].v[j];
	  //printf("j= %d\n", j);
      if (node[target].state == S) {   // == S
		// There is an implicit dt factor (== 1 day)
	    //printf("  An infected found S\n");

		float a = 1. / beta_normal;
		//float a = 2.23;
	    float b = 1. / 0.37;
		float g = gsl_ran_gamma(r_rng, a, b);
#if EXP
	    prob = 1.-exp(-beta[type] * node[source].w[j]);
		//printf("infect prob: %f\n", prob); // 0.09
#else
	    prob = beta[type] * node[source].w[j];
#endif
	  
	    if (gsl_rng_uniform(random_gsl) < prob) {
	      //Check if asymptomatic

	      if (gsl_rng_uniform(random_gsl) < p) {
		    addToList(&new_latent_asymptomatic, target);
			//printf("... add new latent asymptomatic (%d)\n", count_l_symp);
			count_l_asymp += 1;
		  } else {
		    addToList(&new_latent_symptomatic, target);
			count_l_symp += 1;
			//printf("... add new latent symptomatic (%d)\n", count_l_symp);
		  }
	      
	      //Update target data
	      node[target].state = L;

	      //Various
	      n_active++;
	    }
	  } // == S
  } // for  
}

void latency()
{
  int id;

  // prob goes to 1 as eps_S -> 0
  //printf("prob: %f, n= %d\n", 1.-exp(-epsilon_symptomatic), latent_symptomatic.n);
#if 1
  for (int i=0; i < latent_symptomatic.n; i++) {
      //printf("i= %d\n", i);
      id = latent_symptomatic.v[i];  
	  //printf("id of latent_symptomatics: %d\n", id);  // looks ok
#if EXP
	  if (gsl_rng_uniform(random_gsl) < (1.-exp(-epsilon_symptomatic)))
#else
	  if (gsl_rng_uniform(random_gsl) < epsilon_symptomatic)
#endif
	  {
	    //addToList(&new_pre_symptomatic, i); // orig
	    addToList(&new_infectious_symptomatic, id);
		count_i_symp += 1;
		//count_l_presymp += 1;
		//printf("... add new_pre_symptomatic\n");
	    node[id].state = IS;
	    i = removeFromList(&latent_symptomatic, i);
	  } else {
	     //printf("else in epsilon_S\n");
		 ;
	  }
  }
#endif
}


void IsTransition()
{
  int id;

  // Modified by GE to implement simple SIR model. No hospitalizations, ICU, etc.
  for (int i=0; i < infectious_symptomatic.n; i++) {
    id = infectious_symptomatic.v[i];   // id is always zero. BUG
	//printf("id=infectious_symptomatic.v[%d]= %d\n", i, id);
    if (gsl_rng_uniform(random_gsl) < mu) { //days to R/Home
      addToList(&new_recovered, id);
	  //printf("... add recovered\n");
	  count_recov += 1;;
      //printf("node %d -> R\n", id);  // Always zero. BUG. 
      node[id].state = R;
      n_active--;
      i = removeFromList(&infectious_symptomatic, i);
	}
  }
  return;
}


void updateTime()
{ 
  t++;
  //printf("     Update Time, t= %d\n", t);
}

void resetVariables()
{  
  latent_asymptomatic.n = 0;
  latent_symptomatic.n = 0;
  infectious_asymptomatic.n = 0;
  pre_symptomatic.n = 0;
  infectious_symptomatic.n = 0;
  home.n = 0;
  hospital.n = 0;
  icu.n = 0;
  recovered.n = 0;

  for(int i=0;i<NAGE;i++)
    {
      latent_asymptomatic.cum[i] = 0;
      latent_symptomatic.cum[i] = 0;
      infectious_asymptomatic.cum[i] = 0;
      pre_symptomatic.cum[i] = 0;
      infectious_symptomatic.cum[i] = 0;
      home.cum[i] = 0;
      hospital.cum[i] = 0;
      icu.cum[i] = 0;
      recovered.cum[i] = 0;
    }

  t = 0;
  n_active = 0;
}

void resetNodes()
{
  for(int i=0;i<N;i++)
    node[i].state = S;
}

void resetNew()
{
  new_latent_asymptomatic.n = 0;
  new_latent_symptomatic.n = 0;
  new_infectious_asymptomatic.n = 0;
  new_pre_symptomatic.n = 0;
  new_infectious_symptomatic.n = 0;
  new_home.n = 0;
  new_hospital.n = 0;
  new_icu.n = 0;
  new_recovered.n = 0;
}

void updateLists()
{
  updateList(&latent_asymptomatic,&new_latent_asymptomatic);//LA
  updateList(&latent_symptomatic,&new_latent_symptomatic);//LS
  updateList(&infectious_asymptomatic,&new_infectious_asymptomatic);//IA
  updateList(&pre_symptomatic,&new_pre_symptomatic);//PS
  updateList(&infectious_symptomatic,&new_infectious_symptomatic);//IS
  updateList(&home,&new_home); //Home
  updateList(&hospital,&new_hospital);//H
  updateList(&icu,&new_icu);//ICU
  updateList(&recovered,&new_recovered);//R
}

void results(int run)
{
  //Cumulative values
  for(int i=0;i<NAGE;i++)
    fprintf(f_cum,"%d %d %d %d %d %d %d %d %d %d \n",
       latent_asymptomatic.cum[i],
       latent_symptomatic.cum[i],
       infectious_asymptomatic.cum[i],
       pre_symptomatic.cum[i],
       infectious_symptomatic.cum[i],
       home.cum[i],
       hospital.cum[i],
       icu.cum[i],
       recovered.cum[i],
       run);
}
