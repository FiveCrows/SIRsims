#include "head.h"
char folder[80] = "Data_BA";
//char folder[80] = "Data";

void readData()
{
  readNetwork();
  readNodes();
}

void readParameters()
{
  FILE *f;
  char trash[100];

  sprintf(trash, parameter_file, parameters);
  printf("parameter_file= %s\n", parameter_file);
  f = fopen(trash, "r");
  fscanf(f,"%s %d", trash, &N); //Nodes
  fscanf(f,"%s %lf", trash, &r); //relative inf. of asymptomatic individuals
  fscanf(f,"%s %lf", trash, &epsilon_asymptomatic); //asymptomatic latent period-1
  epsilon_asymptomatic = 1.0/epsilon_asymptomatic;
  fscanf(f,"%s %lf", trash, &epsilon_symptomatic); //symptomatic latent period-1
  epsilon_symptomatic = 1.0/epsilon_symptomatic;
  fscanf(f,"%s %lf", trash, &p); //proportion of asymptomatic
  fscanf(f,"%s %lf", trash, &gammita); //pre-symptomatic period
  gammita = 1.0/gammita;
  fscanf(f,"%s %lf", trash, &mu); //time to recover
  mu = 1.0/mu;
  for(int i=0;i<NAGE;i++){ //symptomatic case hospitalization ratio
    fscanf(f,"%s %lf", trash, alpha+i);
    alpha[i] = alpha[i]/100;
  }
  for(int i=0;i<NAGE;i++){
    fscanf(f,"%s %lf", trash, xi+i); //ICU ratio
    xi[i] = xi[i]/100;
  }
  fscanf(f,"%s %lf", trash, &delta); //time to hospital
  delta = 1.0/delta;
  fscanf(f,"%s %lf", trash, &muH); //recovery in hospital
  muH = 1.0/muH;
  fscanf(f,"%s %lf", trash, &muICU); //recovery in ICU
  muICU = 1.0/muICU;
  fscanf(f,"%s %lf", trash, &k); //k
  fscanf(f,"%s %lf", trash, &beta_normal); //infectivity
  printf("fscanf, beta_normal= %lf\n", beta_normal);
  printf("fscanf, r= %lf\n", r);
  fclose(f);
}

void readNetwork()
{
  int s, t;
  double w;
  FILE *f;
  char *token;
  char string[500];
  
  node = malloc(N * sizeof *node);
  for(int i=0;i<N;i++)
    {
      node[i].k = 0;
      node[i].v = malloc(sizeof *node[i].v);
      node[i].w = malloc(sizeof *node[i].w);
    }

  //f = fopen("Data/network.txt","r");
  //f = fopen("Data_SIR/edges_BA.csv", "r");
  printf("network_file= %s\n", network_file);
  f = fopen(network_file, "r");
  while(fgets(string,500,f))
    {
      token = strtok(string," ");
      s = atoi(token);
      token = strtok(NULL," ");
      t = atoi(token);
      token = strtok(NULL,"\n");
      w = atof(token);

      node[s].k++;
      //Update size of vectors
      node[s].v = realloc(node[s].v, node[s].k * sizeof *node[s].v);
      node[s].w = realloc(node[s].w, node[s].k * sizeof *node[s].w);
      //Write data
      node[s].v[node[s].k-1] = t;
      node[s].w[node[s].k-1] = w;

	  // The input data was an undirected graph
	  if (t != s) {
		node[t].k++;
      	node[t].v = realloc(node[t].v, node[t].k * sizeof *node[t].v);
        node[t].w = realloc(node[t].w, node[t].k * sizeof *node[t].w);
        node[t].v[node[t].k-1] = s;
        node[t].w[node[t].k-1] = w;
	  }
    }
  printf("exit while\n");
  fclose(f);
}

void readNodes()
{
  int s, age;
  FILE *f;
  char *token;
  char string[500];
  
  //f = fopen("Data/nodes.txt", "r");
  f = fopen(node_file, "r");
  while(fgets(string,500,f))
    {
      token = strtok(string," ");
      s = atoi(token);
      token = strtok(NULL,"\n");
      age = atoi(token);

      node[s].age = age;
    }
  fclose(f);
}
