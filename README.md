# covid-wave
A SIR+H (H=hospital admissions compartment) model of SARS-CoV2 spread in the UK.

A work in progress.

Performs bayesian inference against observed new daily case and hospital admissions data using PyMC3's NUTS sampler.

Future dev:

- Incorporate policy indices (with associated error terms) in reccurence equations and perhaps a new RV as a scaling factor.
- Alternatively, split the dataset into timespans denoting common stringency levels.
- Account for vaccinations.

		
		



Credit to https://github.com/Priesemann-Group/covid_bayesian_mcmc for the idea of using a theano scan function to implement the recurrence equations.

Data sources: https://coronavirus.data.gov.uk/details/healthcare
