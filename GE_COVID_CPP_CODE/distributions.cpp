// Distributions useful for Covid modeling

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_permutation.h>
#include <string>
#include <vector>

using namespace std;

// Experiment with creating a negative binomial of mean R0 and variance
// R0 * (1 + k*R0)
//
//  http://gnu.ist.utl.pt/software/gsl/manual/html_node/The-Negative-Binomial-Distribution.html
//  gsl_ran_negative_binomial (const gsl_rng * r, double p, double n)
//
//  https://www.gnu.org/software/gsl/doc/html/randist.html#the-gamma-distribution
//  double gsl_ran_gamma(const gsl_rng * r, double a, double b)
//
// https://www.gnu.org/software/gsl/doc/html/randist.html#the-exponential-distribution
// double gsl_ran_exponential(const gsl_rng * r, double mu)
//
// https://www.gnu.org/software/gsl/doc/html/randist.html#the-geometric-distribution
// unsigned int gsl_ran_geometric(const gsl_rng * r, double p)
//
// https://www.gnu.org/software/gsl/doc/html/randist.html?highlight=poisson#c.gsl_ran_poisson
// unsigned int gsl_ran_poisson(const gsl_rng * r, double mu)

void mean_var(string msg, vector<double>& samples, int n) 
{
	double mean = gsl_stats_mean(&samples[0], 1, n);
	double var = gsl_stats_variance(&samples[0], 1, n);
	printf("(%s) mean= %f, var= %f\n", msg.c_str(), mean, var);
}

void gamma(vector<double>& samples, gsl_rng* r_rng, double R0, double k, int n)
{
	// Assumes that samples are pre-allocated
	double alpha = k;
	double beta = R0 / k;
    printf("dist.gamma: alpha, beta= %f, %f\n", alpha, beta);

	for (int i=0; i < n; i++) {
		samples[i] = gsl_ran_gamma(r_rng, alpha, beta);
	}
}

void poisson(vector<double>& samples, gsl_rng* r_rng, double lmbda, int n)
{
	for (int i=0; i < n; i++) {
		samples[i] = gsl_ran_poisson(r_rng, lmbda);
	}
}

void poisson(vector<double>& samples, gsl_rng* r_rng, vector<double>& lmbda, int n)
{
	for (int i=0; i < n; i++) {
		samples[i] = gsl_ran_poisson(r_rng, lmbda[i]);
	}
}

void exponential(vector<double>& samples, gsl_rng* r_rng, double lmbda, int n)
{
	for (int i=0; i < n; i++) {
    	samples[i] = gsl_ran_exponential(r_rng, lmbda);
	}
}

void negativeBinomial(vector<double>& samples, gsl_rng* r_rng, double R0, double k, int n)
{
    // average mu=R0, var=sigma**2=mu*(1+k*mu)
    // k: Dispersion
    // R0: reproduction number at early times
    double p = 1 / (1+R0/k);
    //n = 1. / k  # do not know which is correct. This or next line
    double r = k;
	for (int i=0; i < n; i++) {
    	samples[i] = gsl_ran_negative_binomial(r_rng, p, r);
	}
}

void weibull(vector<double>& samples, gsl_rng* r_rng, double shape, double scale, int n)
{
	for (int i=0; i < n; i++) {
    	samples[i] = gsl_ran_weibull(r_rng, scale, shape);
	}
}
//------------------------------------------------------
int main()
{
	const gsl_rng_type* T;
	gsl_rng* r_rng;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r_rng = gsl_rng_alloc(T);

	vector<double> samples;
	int N = 10000;
	samples.resize(N);

	gamma(samples, r_rng, 2., .2, N);
	mean_var("gamma", samples, N);

	poisson(samples, r_rng, 3., N);
	mean_var("Poisson", samples, N);

	exponential(samples, r_rng, 3., N);
	mean_var("Exponential", samples, N);

	negativeBinomial(samples, r_rng, 3., 0.75, N);
	mean_var("Negative Binomial", samples, N);

	double R0 = 2.5;
	double k = 0.75;

	N = 200000;
	double p = 1. / (1+R0/k);
	double r =  k;
	samples.resize(N);

  	int seed = gsl_rng_uniform_int(r_rng, 10000);
	printf("seed= %d\n", seed);
	negativeBinomial(samples, r_rng, R0, k, N);
	mean_var("- Negative Binomial", samples, N);
	printf("p= %f, r= %f\n", p, r);
	printf("k= %f\n", k);
	printf("R0= %f,    R0*(1.+R0/k)= %f\n", R0, R0*(1.+R0/k));

	// ----------------------------------------------
	string strg = "\n\
==============================================\n\
POISSON MIXTURE WITH GAMMA is a NEGATIVE BINOMIAL\n\
Choose R according to Poisson with mean R0\n\
Sample Poisson with this R\n\
==============================================\
";
	printf("%s\n", strg.c_str());

	gamma(samples, r_rng, R0, k, N);
	mean_var("gamma", samples, N);

	poisson(samples, r_rng, samples, N);
	mean_var("gamma-poisson", samples, N);

	double shape = 2.;
	double scale = 3.;
	weibull(samples, r_rng, shape, scale, N);
	mean_var("weibull", samples, N);

