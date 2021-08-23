import theano.tensor as tt
import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
#matplotlib inline
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import scipy.stats as stats

# Read UK hospitalisations data from csv file
# Make one count_data table for each home nation. England, NI, Scotland, Wales.
# Sort into date order
# Take a n_count_data from the number of dates in each nation table.
raw_data = pd.read_csv("data/data_2021-Jun-17.csv")
count_data = [raw_data[['newAdmissions', 'date']][raw_data['areaName']== \
    pd.unique(raw_data['areaName'])[i]] for i in range(0,4)]
count_data = [count_data[i].sort_values(by='date' , axis='index') for i in range(0,4)]
n_count_data = [len(count_data[i]) for i in range(0,4)]

# Starting with just hospitalisations within England
n_count_data = n_count_data[0]

# Plots to explore data
figsize(11, 9)
ax = plt.subplot(211)
plt.bar(x=count_data[0]['date'], height=count_data[0]['newAdmissions'])
ax.xaxis.set_major_locator(plt.MaxNLocator(8)) # Sets number of dates shown on x-axis to 8
ax = plt.subplot(212)
plt.hist(count_data[0]['newAdmissions'], histtype='stepfilled', bins=30, alpha=0.85,
         label="new hospital admissions histogram", density=False)
plt.show()

"""MODELLING
"""
# Assign parameters to model. Alpha is defined as 1/mean because 
# that is the expected value of an exponential distribution.When the sampling begins, 
# it will pick a number (n_lambda) of random values from the exponential function parametrised by the 
# same alpha value (one for each lambda). There are multiple lambdas to allow different 
# distributions for different time periods. Tau is the time value used to separate the 
# lambda distributions (to define a change in hospitalisation rate).

n_lambda = 2 #i.e. 6 lambdas (0 to 5 in a shape array) and 5 taus (0 to 4)
j=n_lambda-2
tau = ['']*(n_lambda-1)
with pm.Model() as model:
    alpha = 1.0/count_data[0]['newAdmissions'].mean()
    lambda_array = pm.Exponential("lambda_array", alpha, shape=n_lambda)   
    for i_tau in range(0,n_lambda-1):    # Tau can't be defined like 
        tau[i_tau] = pm.DiscreteUniform("tau_"+str(i_tau), lower= 0 if i_tau==0 else tau[i_tau-1] , upper=n_count_data)

def switch_code2(n_lambda):
    """ Code to allocate the correct variable within lambda_array to lambda_ acccording to which 'tau region' that idx finds
    itself in. Writes out all the switchpoint condition code in a single statement, which needs to be done dynamically as the
    chosen n_lambda value affects the conditional statement.
    """
    code_fragment=[None]*(n_lambda-1)
    for k in range(0,n_lambda-1):
        code_fragment[k] = 'pm.math.switch(idx < tau[j-(j-'+str(k)+')] , lambda_array[j-(j-'+str(k)+')],'
    switch_code2 ='lambda_ = '+' '.join(element) +'lambda_array[j-(j-'+ str(n_lambda-1) +')]'+ ((n_lambda-1)*')')
    return switch_code2

switch_code2 = switch_code2(n_lambda)

with model:
    idx = np.arange(n_count_data)
    exec(switch_code2) #Updated to run on the individually defined taus.

# Use a Poisson distribution to define the combined lambda (appropriate because positive and 
# discrete for the number of hospitalisations in a day). The sampler will compare the values 
# output from the current parameters against the actual data and adjust the parameters in the
# chain as necessary.
with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data[0]['newAdmissions'].values)

with model:
    step = pm.Metropolis()
    trace = pm.sample(30000, tune = 2000, step=step) #tune adjusts step-size, these samples should be discarded
    lambda_samples = trace['lambda_array']
    tau_samples = [trace['tau_'+str(i_tau)] for i_tau in range(0, n_lambda-1)]
    pm.traceplot(trace, trace.varnames[1::])


"""PLOTTING
"""
figsize(12.5, 10)
lambda_samples = lambda_samples.T

#histograms for the posterior distributions of lambda and tau:
ax = plt.subplot(311)
plt.hist(lambda_samples[0], histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_0$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_array,\;\tau_array$""")
plt.xlabel("$\lambda_0$ value")

ax = plt.subplot(312)
plt.hist(lambda_samples[1], histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlabel("$\lambda_1$ value")

plt.subplot(313)
w = 1.0 / len(tau_samples[0])* np.ones_like(tau_samples[0]) #rescaling sample frequency to max=1
plt.hist(tau_samples[0], bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$0",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.xlabel(r"$\tau_0$ (in days)")
plt.ylabel("probability")
plt.show()


# Produce the expected admissions per day for England over the course of the pandemic by taking the average of all \
# the samples for lambda_
lambda_samples_d = {}
tau_samples_d = {}
for i_lam in range(0,n_lambda):
    lambda_samples_d[i_lam] = lambda_samples[i_lam]
for i_tau in range(0,n_lambda-1):
    tau_samples_d[i_tau] = tau_samples[i_tau]

expected_admissions_per_day = np.zeros(n_count_data)

def is_day_less_than_tau(day, lambda_samples_ref):
    is_day_less_than_tau = day < tau_samples_d[lambda_samples_ref] if \
        lambda_samples_ref <= max(tau_samples_d.keys()) else \
            day >= tau_samples_d[max(tau_samples_d.keys())]
    return is_day_less_than_tau

# Expected hospital admissions per day. Set a range: 1 element per day. Then measure if day is lower than tau switchpoint\
# value and produce the corresponding lambda value (summated for all 40000 samples)    
for day in range(0,n_count_data):
    lambda_samples_sum_array = [ lambda_samples_d[lambda_samples_ref][is_day_less_than_tau(day,lambda_samples_ref)].sum()\
            for idx,lambda_samples_ref in enumerate(sorted(lambda_samples_d.keys())) ]
    first_nonzero_val = next((value for index,value in enumerate(lambda_samples_sum_array) if value != 0), None)
    expected_admissions_per_day[day] = first_nonzero_val / tau_samples[0].shape[0]

# Plot of the expected hospital admissions per day over the course of the pandemic in England using the posterior \
# distributions of lambda and tau.
ax = plt.subplot(111)
plt.plot(range(n_count_data), expected_admissions_per_day, lw=4, label='expected number of admissions', color="#000000")
plt.bar(x=count_data[0]['date'], height=count_data[0]['newAdmissions'], label="observed admissions per day")
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
plt.legend(loc="upper left")
plt.show()

print("And now, the end is near.")