# covid-wave
A SIR+H (H=hospital admissions compartment) model of SARS-CoV2 spread in the UK.

A work in progress.

Performs bayesian inference against observed new daily case and hospital admissions data using PyMC3's NUTS sampler.

Transmission control measures are incorporated using policy index data from Oxford COVID-19 Government Response Tracker. Various models of its relationship to infection rate parameters are tested.


Credit to https://github.com/Priesemann-Group/covid_bayesian_mcmc for the idea of using a theano scan function to implement the recurrence equations.

Daily cases and hospital admissions: https://coronavirus.data.gov.uk/details/healthcare

UK policy indices: https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/OxCGRT_latest.csv


Future dev:

- Consider non-parametric implementation.
- Account for vaccinations.
