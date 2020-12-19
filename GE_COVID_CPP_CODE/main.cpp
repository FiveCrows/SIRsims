#include "head.h"
#include "G.h"


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

  G* g = new G(params, files, lists, counts, network, gsl);

  g->initialize(argc,argv, files, params, lists, network, gsl);

  g->openFiles(files);
  g->runSimulation(params, lists, counts, network, gsl, files);
  g->closeFiles(files);

  g->print(t0);
  g->freeMemory(params, network, lists, gsl);

  return 0;
}

