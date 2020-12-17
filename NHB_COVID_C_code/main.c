#include "head.h"

int main(int argc, char *argv[])
{
  int t0;
  t0 = time(NULL);

  initialize(argc,argv);

  openFiles();
  runSimulation();
  closeFiles();

  print(t0);

  freeMemory();

  return 0;
}

void initialize(int argc, char *argv[])
{
  int seed;

  n_runs = 100;
  n_runs = 10;
  n_runs = 5;
  parameters = 0;

  seed = time(NULL);
  initRandom(seed);

  // the Results folder contains parameters.txt, result files
  // the Data folder contains network data
  // executable data_folder result_folder

  if (argc != 3) {
	printf("argc= %d\n", argc);
	exit(1);
  }

  strcpy(data_folder, argv[1]);
  strcat(data_folder, "/");
  strcpy(result_folder, argv[2]);
  strcat(result_folder, "/");
  //printf("result_folder: %s\n", result_folder); 
  //printf("data_folder: %s\n", data_folder); 

  //char* filenm = "parameters_0.txt";
  strcpy(parameter_file, result_folder);
  strcat(parameter_file, "parameters.txt");
  printf("parameter_file= %s\n", parameter_file); 
  readParameters(parameter_file);
  allocateMemory();
  readData();
  setBeta();
}

void runSimulation()
{
  for (int run=0;run<n_runs;run++) {
     printf("run= %d\n", run);
     {
  		count_l_asymp   = 0;
  		count_l_symp    = 0;
  		count_l_presymp = 0;
  		count_i_symp    = 0;
  		count_recov     = 0;

        init();

        while(n_active>0) {
          //printf("runSinm n_active= %d\n", n_active);
	      spread(run);
		}

  		printf("total number latent symp:  %d\n", count_l_symp);
  		printf("total number latent presymp:  %d\n", count_l_presymp);
  		printf("total number infectious symp:  %d\n", count_i_symp);
  		printf("total number recovered:  %d\n", count_recov);
        results(run);
     }
  }
}
