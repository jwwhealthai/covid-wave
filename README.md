# covid-wave
A SIR+H (H=hospital admissions compartment) model of UK SARS-CoV2 spread in the UK.

A work in progress.

Performs bayesian inference against observed new daily case and hospital admissions data using PyMC3's NUTS (v slow but more accurate) or Metropolis (faster) samplers.

Future dev:

- Determine cause of gradient error during sampling with NUTS + theano loop.
- Add switchpoints to SIRH model to extend inference across longer daterange.
- Account for vaccinations.

		
		



Credit to https://github.com/Priesemann-Group/covid_bayesian_mcmc for the idea of using a theano scan function to implement the recurrence equations.

Data sources: https://coronavirus.data.gov.uk/details/healthcare
