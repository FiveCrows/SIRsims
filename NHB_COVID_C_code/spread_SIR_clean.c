#include "head.h"

#define EXP 1

// Implement an SIR version of this code. 

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
    //addToList(&latent_symptomatic, seed); // orig (should be new_latent_symptomatic)
    addToList(&infectious_symptomatic, seed); // GE (go directly to IS state
  }

  gsl_permutation_free(p);

  n_active = ninfected;
  count_i_symp += ninfected;
  //printf("added %d latent_symptomatic individuals\n", ninfected);
  printf("added %d infectious_symptomatic individuals\n", ninfected);

  t = 1;
}

void spread(int run)
{  
  resetNew();

  //S to L
  //printf("--> enter infection\n");
  infection();
  //L to P
  //printf("--> enter latency\n");
  //latency();
  //P to I
  //printf("--> enter preToI\n");
  //preToI();
  //printf("--> enter isTransition\n");
  IsTransition();


  updateLists();
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

  for (int i=0; i < pre_symptomatic.n; i++) {
	//printf("call infect(pre_symptomatic) %d/%d\n", i, pre_symptomatic.n);
    infect(pre_symptomatic.v[i], PS);
  }
    
  //Infectious symptomatic
  for (int i=0; i < infectious_symptomatic.n; i++) {
	//printf("call infect(infectious_symptomatic) %d/%d\n", i, infectious_symptomatic.n);
    infect(infectious_symptomatic.v[i], IS);
  }
}

void infect(int source, int type)
{
  int target;
  double prob;
 
  for (int j=0; j < node[source].k; j++) { // for
      target = node[source].v[j];
      if (node[target].state == S) {   // == S
		// There is an implicit dt factor (== 1 day)

		float a = 1. / beta_normal;
		//float a = 2.23;
	    float b = 1. / 0.37;
		float g = gsl_ran_gamma(r_rng, a, b);
#if EXP
	    prob = 1.-exp(-beta[type] * node[source].w[j]);
#else
	    prob = beta[type] * node[source].w[j];
#endif
	  
	    if (gsl_rng_uniform(random_gsl) < prob) {
	      //Check if asymptomatic

	      if (gsl_rng_uniform(random_gsl) < p) {
		    addToList(&new_latent_asymptomatic, target);
			count_l_asymp += 1;
			printf("... add new latent asymptomatic (%d, %d)\n", j, count_l_asymp);
			exit(1);
		  } else {
		    //addToList(&new_latent_symptomatic, target); // orig code
		    addToList(&new_infectious_symptomatic, target);  // GE: infectious to infectious
			count_i_symp += 1;
			//printf("... add new infectious symptomatic (%d, %d)\n", j, count_i_symp);
			//printf("... add new latent symptomatic (%d, %d)\n", j, count_l_symp);
		  }
	      
	      //Update target data
	      node[target].state = IS;

	      //Various
	      n_active++;
	    }
	  } // == S
  } // for  
}

void latency()
{
  if (latent_asymptomatic.n > 0) {printf("latent_asymptomatic should be == 0\n"); exit(1); }
  if (latent_symptomatic.n > 0) {printf("latent_symptomatic should be == 0\n"); exit(1); }

#if 0
  int id;

#if 1
  for (int i=0; i < latent_symptomatic.n; i++) {
      id = latent_symptomatic.v[i];
#if EXP
	  if (gsl_rng_uniform(random_gsl) < (1.-exp(-epsilon_symptomatic)))
#else
	  if (gsl_rng_uniform(random_gsl) < epsilon_symptomatic)
#endif
	  {
	    //addToList(&new_pre_symptomatic, id);  // orig code (had i instead of id)
	    addToList(&new_infectious_symptomatic, id); // GE (
		count_l_presymp += 1;
		//printf("... add new_pre_symptomatic (%d, %d, %d)\n", i, count_l_presymp, latent_symptomatic.n);
	    node[id].state = PS;
	    i = removeFromList(&latent_symptomatic, i);
	  } else {
	     ; //printf("else in epsilon_S\n");
	  }
  }
#endif

#endif
}

void preToI()
{
  int id;

  for (int i=0; i < pre_symptomatic.n; i++) {
      id = pre_symptomatic.v[i];
#if EXP
      if (gsl_rng_uniform(random_gsl) < (1.-exp(-gammita))) { //onset of symptoms
#else
      if (gsl_rng_uniform(random_gsl) < gammita) { //onset of symptoms
#endif
	    addToList(&new_infectious_symptomatic, id);
		count_i_symp += 1;
		//printf("... add new infectious_symptomatic from pre_S (%d, %d)\n", i, count_i_symp);
	    node[id].state = IS;

	    i = removeFromList(&pre_symptomatic, i);
	  }
  }
}

void IsTransition()
{
  int id;

  // Modified by GE to implement simple SIR model. No hospitalizations, ICU, etc.
  for (int i=0; i < infectious_symptomatic.n; i++) {
    id = infectious_symptomatic.v[i];
    if (gsl_rng_uniform(random_gsl) < mu) { //days to R/Home
      addToList(&new_recovered, id);
	  count_recov += 1;;
	  //printf("... add recovered (%d, %d)\n", i, count_recov);
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
