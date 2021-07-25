#import aesara as ae
#import aesara.tensor as aet
#import theano-pymc as tt
import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
#from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import scipy.stats as stats
import sunode
import sunode.wrappers.as_theano
import theano



"""
---------------------------------------------------------------------------------
Define functions
---------------------------------------------------------------------------------
"""
# Data extraction function:
def COVID_data(daterange):
    raw_data = pd.read_csv("data/cases/data_2021-Jul-13.csv") # 'newCasesByPublishDate'
    raw_data['date_new'] = pd.to_datetime(raw_data['date'])
    raw_data = raw_data.sort_values(by='date_new' , axis='index')
    cases = raw_data[['newCasesByPublishDate','date_new']]
    return cases


# Modelfunction defined according to sunode conventions:
def SIR_sunode(t, y, p):
    return {
        'S': -p.beta * y.S * y.I,
        'I': p.beta * y.S * y.I - p.lam * y.I}

"""
---------------------------------------------------------------------------------
Configuration settings, parameter priors etc.
---------------------------------------------------------------------------------
"""
n_samples = 500
n_tune = 100
daterange = pd.date_range(start="2020-08-26", end="2020-11-13")
covid_obj = COVID_data(daterange)                                       # Make a function to extract case data when given a country code argument

N = 67000000                                                        # Population of UK
I_init_date_mask = (covid_obj['date_new'] > (daterange[0]-11*daterange.freq)) & (covid_obj['date_new'] <= daterange[0]) # Past 10 days of cases considered to be infectious
I_start_est = np.sum(covid_obj.loc[I_init_date_mask]['newCasesByPublishDate'])       # Initial number of infected at the start of the wave (take as case numbers for the prior 7 days to 26 Aug)
not_S_date_mask = (covid_obj['date_new']<= daterange[0])                           # Assumes all prior cases are immune, deceased or infectious
S_start_est = N - np.sum(covid_obj.loc[not_S_date_mask]['newCasesByPublishDate'])   # S(0) can be estimated as N - I(0) i.e. total number of people minus those infected at the start of the pandemic
wave_date_mask = (covid_obj['date_new'] >= daterange[0]) & (covid_obj['date_new'] <= daterange[-1])
cases_obs = covid_obj.loc[wave_date_mask]['newCasesByPublishDate']
cases_obs_scaled = [((x - np.average(cases_obs)) / np.std(cases_obs)) for x in cases_obs]   # Standardises daily case counts for the wave
I_start_est_scaled = (I_start_est - np.average(cases_obs)) / np.std(cases_obs)
S_start_est_scaled = (S_start_est - np.average(cases_obs)) / np.std(cases_obs)
likelihood = {'distribution': 'normal',
                'sigma': 1}     # Is this valid?
prior = {'beta': -1.5,
            'beta_std': 1,
            'lam': -1.5,
            'lam_std': 1,
            'S_init_mu': S_start_est_scaled,
            # 'S_init_mu': 1,
            'I_init_mu': I_start_est_scaled     # Both of these use the scale set by the observed case numbers
        }


"""
----------------------------------------------------------------------------------
Extract case data for UK, this will be the observed data for Diagnosed compartment
----------------------------------------------------------------------------------
"""

# count_data = [raw_data[['newAdmissions', 'date']][raw_data['areaName']== \
#     pd.unique(raw_data['areaName'])[i]] for i in range(0,4)]
# count_data = [count_data[i].sort_values(by='date' , axis='index') for i in range(0,4)]
# n_count_data = [len(count_data[i]) for i in range(0,4)]


"""
----------------------------------------------------------------------------------
Perform SIDARTHE modelling on waves 1 and 2
----------------------------------------------------------------------------------
"""
with pm.Model() as model:
    sigma = pm.HalfCauchy('sigma', likelihood['sigma'], shape=1)
    beta = pm.Lognormal('beta', prior['beta'], prior['beta_std'])       # lognormal might not be appropriate
    lam = pm.Lognormal('lambda', prior['lam'], prior['lam_std'])
    # S_init_mu = pm.DiscreteUniform('S_init_mu', prior['S_init_mu']*0.99, prior['S_init_mu']*1.01)
    S_init_mu = pm.Uniform('S_init_mu', -1,1)
    # S_init_std = pm.Uniform('S_init_std', 0, prior['S_init_mu']*0.02)
    S_init_std = pm.Constant('S_init_std', 1)
    S_init = pm.Normal('S_init', mu=S_init_mu, sigma=S_init_std)
    # I_init_std = pm.Uniform('I_init_std', prior['I_init_mu']*3, prior['I_init_mu']*2.4)
    I_init_std = pm.Constant('I_init_std', 1)
    # I_init_mu = pm.DiscreteUniform('I_init_mu', prior['I_init_mu']*1.2, prior['I_init_mu']*0.8)
    I_init_mu = pm.Uniform('I_init_mu', -1, 1)

    I_init = pm.Normal('I_init', mu=I_init_mu, sigma=I_init_std)

    
    # ae.mode='DebugMode'

    res, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
        'S': (S_init, ()), 'I': (I_init, ()),},  
        params={
        'beta': (beta, ()), 'lam': (lam, ()), '_dummy': (np.array(1.), ())},
        rhs=SIR_sunode,
        tvals=daterange.dayofyear,
        t0=daterange.dayofyear[0]
    )

    #if(likelihood['distribution'] == 'lognormal'):
    #    I = pm.Lognormal('I', mu=res['I'], sigma=sigma, observed=cases_obs_scaled)
    #elif(likelihood['distribution'] == 'normal'):
    I = pm.Normal('I', mu=res['I'], sigma=0.25, observed=cases_obs_scaled)
    
    R0 = pm.Deterministic('R0',beta/lam)

    trace = pm.sample(n_samples, tune=n_tune, cores=1
    # mode='DebugMode', 
    # start={
    #     'sigma': np.array([0.]), 
    #     'beta': np.array([0.]), 
    #     'lambda': np.array([0.]), 
    #     'S_init_mu': np.array([0.]),
    #     'S_init_std': np.array([0.]),
    #     'S_init': np.array([0.]),
    #     'I_init_mu': np.array([0.]),
    #     'I_init_std': np.array([0.]),
    #     'I_init': np.array([0.]) 
    #     }
    )
    az.plot_trace(trace)
plt.show()
print('this is the end')
"""
----------------------------------------------------------------------------------

----------------------------------------------------------------------------------
"""