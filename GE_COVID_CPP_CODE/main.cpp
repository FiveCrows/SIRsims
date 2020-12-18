#include "head.h"
#include "G.h"


int main(int argc, char *argv[])
{
G g;

  int t0;
  t0 = time(NULL);
  Params params;
  Files files;
  Lists lists;
  Network network;
  GSL gsl;

  g.initialize(argc,argv, files, params, lists, network);

  g.openFiles(files);
  g.runSimulation(params, lists);
  g.closeFiles(files);

  g.print(t0);
  g.freeMemory(params, network, lists, gsl);

  return 0;
}

