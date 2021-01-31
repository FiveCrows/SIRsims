#include "head.h"
#include "G.h"
#include "cxxopts.hpp"
#include <gsl/gsl_rng.h>

#if 0
void parse(int argc, char* argv[])
{
	try {
        cxxopts::Options options(argv[0], " - example command line options");
        options
          .positional_help("[optional args]")
          .show_positional_help();
    
        options
          .allow_unrecognised_options()
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

    auto result = options.parse(argc, argv);
    auto arguments = result.arguments();
    } catch(const cxxopts::OptionException& e) {
    		std::cout << "error parsing options: " << e.what() << std::endl;
    		exit(1);
	}
}
#endif


int main(int argc, char *argv[])
{

  int t0;
  t0 = time(NULL);
  Params params;
  Files files;
  Lists lists;
  Counts counts;
  Network network;
  GSL gsl;


  gsl_rng_env_setup();
  gsl_rng * r;  /* global generator */
  const gsl_rng_type * T;
  T =  gsl_rng_default;
  r = gsl_rng_alloc (T);

  printf ("generator type: %s\n", gsl_rng_name (r));
  printf ("seed = %lu\n", gsl_rng_default_seed);
  printf ("first value = %lu\n", gsl_rng_get (r));

  gsl_rng_free (r);
  
  

  G* g = new G(params, files, lists, counts, network, gsl);
  g->initialize(argc,argv, files, params, lists, network, gsl);

  g->openFiles(files);
  g->runSimulation(params, lists, counts, network, gsl, files);
  g->closeFiles(files);

  g->print(t0);
  g->freeMemory(params, network, lists, gsl);

  return 0;
}

