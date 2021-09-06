import aesara as ae
import aesara.tensor as aet
#import theano-pymc as tt
import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
#from IPython.core.pylabtools import figsize
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import scipy.stats as stats
import sunode
import sunode.wrappers.as_theano
import cloudpickle
#import theano
#import theano.tensor as tt
import datetime as dt
import multiprocessing as mp

# mp.set_start_method('fork')
"""
---------------------------------------------------------------------------------
Define data wrangling functions
---------------------------------------------------------------------------------
"""
# Data extraction function:
def COVID_case_data(daterange):
    raw_data_c = pd.read_csv("data/cases/data_2021-Jul-13.csv") # 'newCasesByPublishDate'
    raw_data_c['date_new'] = pd.to_datetime(raw_data_c['date'])         # coerces date column to datetime
    raw_data_c = raw_data_c.sort_values(by='date_new' , axis='index')   # sorts so that earliest date appears first
    cases = raw_data_c[['newCasesByPublishDate','date_new']]          # makes dataframe of date and cases columns
    cases = cases.reset_index()
    return cases

def COVID_hosp_data(daterange):
    raw_data_h = pd.read_csv("data/hosp/data_2021-Jun-17.csv") # 'newAdmissions'
    raw_data_h['date_new'] = pd.to_datetime(raw_data_h['date'])         # coerces date column to datetime
    raw_data_h = raw_data_h.sort_values(by='date_new' , axis='index')   # sorts so that earliest date appears first
    tmp_data_h = raw_data_h.groupby(by=[raw_data_h.date_new], as_index=False).sum()     # groups 4 nations data by date and sums admissions
    admissions = tmp_data_h[['newAdmissions','date_new']]          # makes dataframe of date and admissions columns
    admissions = admissions.reset_index()
    return admissions

def infectious_duration(tau):
    # if isinstance(tau, (int,float)):
    #     res = tau.get_test_value()
    # else:
    res = tau
    return res

def get_initial_I(tau, covid_case_obj, daterange):
    I_initial = np.sum(covid_case_obj.iloc[(covid_case_obj.index[covid_case_obj.date_new==daterange[0]].values[0] - infectious_duration(tau)) : covid_case_obj.index[covid_case_obj.date_new==daterange[0]].values[0]+1]['newCasesByPublishDate'])
    # I0 = pd.Series(I_initial)
    # I0.name = 'I0'
    # I0 = covid_case_obj.loc[(covid_case_obj['date_new'] > starttime - dt.timedelta(days=tau)) & (covid_case_obj['date_new'] <= starttime)]['newCasesByPublishDate']
    return I_initial

def get_initial_H(tau, covid_hosp_obj, daterange):
    H_initial = np.sum(covid_hosp_obj.iloc[(covid_hosp_obj.index[covid_hosp_obj.date_new==daterange[0]].values[0] - infectious_duration(tau)) : covid_hosp_obj.index[covid_hosp_obj.date_new==daterange[0]].values[0]+1]['newAdmissions'])
    return H_initial

def get_cases_obs(tau, covid_case_obj, dataset, daterange):
    if dataset == 'cases':
        column = 'newCasesByPublishDate'
    elif dataset == 'admissions':
        column = 'newAdmissions'
    cases_obs = covid_case_obj.iloc[ \
                (covid_case_obj.index[covid_case_obj.date_new == daterange[0]].values[0] - infectious_duration(tau)) : covid_case_obj.index[covid_case_obj.date_new == daterange[-1]].values[0]+1] \
                    [column]
    cases_obs.name = '{}_obs'.format(dataset)
    return cases_obs

def get_cases_obs_movtautot(tau, covid_case_obj, dataset, daterange):
    cases_obs = get_cases_obs(tau, covid_case_obj, dataset, daterange)
    cases_obs_tmp = np.trim_zeros( \
            [np.sum(cases_obs[idx-infectious_duration(tau):idx+1]) for idx,i in enumerate(cases_obs)] \
                , trim='f')
    # cases_obs_movtautot = pd.Series(cases_obs_tmp)
    cases_obs_movtautot = pd.Series(cases_obs_tmp)
    cases_obs_movtautot.name = '{}_obs_movtautot'.format(dataset)
    return cases_obs_movtautot

"""
---------------------------------------------------------------------------------
ODE system which defines the model (SIR+H)
---------------------------------------------------------------------------------
"""
# Modelfunction defined according to sunode conventions (where LHS labels actually represent the first order derivatives)
def SIR_sunode(t, y, p):
    return {
        'S': -p.beta * y.S * y.I,               #Term1: Change due to new infections NB: assumes all infected become immune, otherwise wold require another term
        'I': p.beta * y.S * y.I - p.lam * y.I - p.gamma * y.I,  #Term1: Change due to new infections; Term2: Change due to those removed without requiring hospital; Term3: Change due to those moving to the hospital compartment
        'H': p.gamma * y.I - p.delta * y.H      #Term1: Change due to new hospitalisations; Term2: Change due to 'removed' from hospital compartment (deceased or recovered)
        }
"""
---------------------------------------------------------------------------------
Configuration settings, parameter priors etc.
---------------------------------------------------------------------------------
"""
n_samples = 800
n_tune = 200
daterange = pd.date_range(start="2020-08-26", end="2020-11-13")
covid_case_obj = COVID_case_data(daterange)                                       # Make a function to extract case data when given a country code argument
covid_hosp_obj = COVID_hosp_data(daterange)
# endtime = dt.date(2020,11,13)
# starttime = dt.date(2020,8,26)
# test_tau = dt.timedelta(days=10)

N = 67000000                                                        # Population of UK
tau = 10                                                          # days of infectiousness
I_initial = get_initial_I(tau, covid_case_obj, daterange) /N
S_initial = (N - np.sum(covid_case_obj.loc[covid_case_obj['date_new']<= daterange[0]]['newCasesByPublishDate']) )/N
H_initial = 1139 / N         # 1139 is the sum of the previous 10days of hospital admissions
cases_obs = get_cases_obs_movtautot(tau, covid_case_obj, dataset='cases', daterange=daterange) /N
hosp_admissions_obs = get_cases_obs_movtautot(tau, covid_hosp_obj, dataset='admissions', daterange=daterange) /N

# I_init_date_mask = (covid_case_obj['date_new'] > (daterange[0]-11*daterange.freq)) & (covid_case_obj['date_new'] <= daterange[0]) # Past 10 days of cases considered to be infectious
# I_start_est = np.sum(covid_case_obj.loc[I_init_date_mask]['newCasesByPublishDate'])       # Initial number of infected at the start of the wave (take as case numbers for the prior 7 days to 26 Aug)
# not_S_date_mask = (covid_case_obj['date_new']<= daterange[0])                           # Assumes all prior cases are immune, deceased or infectious
# S_start_est = N - np.sum(covid_case_obj.loc[not_S_date_mask]['newCasesByPublishDate'])   # S(0) can be estimated as N - I(0) i.e. total number of people minus those infected at the start of the pandemic
# wave_date_mask = (covid_case_obj['date_new'] >= daterange[0]) & (covid_case_obj['date_new'] <= daterange[-1])
# cases_obs = covid_case_obj.loc[wave_date_mask]['newCasesByPublishDate']
# cases_obs_scaled = [((x - np.average(cases_obs)) / np.std(cases_obs)) for x in cases_obs]   # Standardises daily case counts for the wave
# I_start_est_scaled = (I_start_est - np.average(cases_obs)) / np.std(cases_obs)
# S_start_est_scaled = (S_start_est - np.average(cases_obs)) / np.std(cases_obs)
likelihood = {'distribution': 'normal',
                'sigma': 1.0,
                'sigma_h': 1.0}
                # 'H_sigma': 1.0}     # Is this valid?.. try with one likelihood parameter
prior = {'beta_mu': 1.5,
            'beta_sig': 0.25,
            'gamma_mu': 0.10,
            'gamma_sig': 0.25,
            'delta_mu': 0.005,
            'delta_sig': 0.25,
            'lam_mu': 1.0,
            'lam_sig': 0.25
            # 'S_init_mu': S_start_est_scaled,
            # 'S_init_mu': 1,
            # 'I_init_mu': I_start_est_scaled     # Both of these use the scale set by the observed case numbers
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
    # tau = pm.DiscreteUniform('tau', 7, 15)
    # I_initial = pm.Deterministic('I_initial', get_initial_I(tau, covid_case_obj)/N) # Past tau days of cases considered to be infectious
    # cases_obs = pm.Deterministic('cases_obs', get_cases_obs(tau, covid_case_obj)/N) 
    # cases_obs = pm.Deterministic('cases_obs', cases_obs_ct / N)
    # testing_movtautot = get_cases_obs_movtautot(tau, covid_case_obj)
    # cases_obs_movtautot = pm.Deterministic('cases_obs_movtautot', get_cases_obs_movtautot(tau, covid_case_obj)/N)

    sigma = pm.HalfCauchy('sigma', likelihood['sigma'])
    sigma_h = pm.HalfCauchy('sigma_h', likelihood['sigma_h'])
    # H_sigma = pm.HalfCauchy('H_sigma', likelihood['H_sigma'])
    beta = pm.Lognormal('beta', prior['beta_mu'], prior['beta_sig'])     # lognormal might not be appropriate
    lam = pm.Lognormal('lambda', prior['lam_mu'], prior['lam_sig'])
    gamma = pm.Lognormal('gamma', prior['gamma_mu'], prior['gamma_sig'])
    delta = pm.Lognormal('delta', prior['delta_mu'], prior['delta_sig'])

    #I = pm.Normal('I', mu=res['I'], sigma=0.25)
    # new_cases_t-tau = 
    # new_cases_today = I - new_cases_yesterday - new_cases_daybefore - ... new_cases_taudaysago
    

    # S_init_mu = pm.DiscreteUniform('S_init_mu', prior['S_init_mu']*0.99, prior['S_init_mu']*1.01)
    # S_init_mu = pm.Uniform('S_init_mu', 0.999,1)
    # S_init_std = pm.Uniform('S_init_std', 0, prior['S_init_mu']*0.02)
    # S_init_std = pm.Constant('S_init_std', 1)
    # S_init = pm.Normal('S_init', mu=S_init_mu, sigma=S_init_std)
    # I_init_std = pm.Uniform('I_init_std', prior['I_init_mu']*3, prior['I_init_mu']*2.4)
    # I_init_std = pm.Constant('I_init_std', 1)
    # I_init_mu = pm.DiscreteUniform('I_init_mu', prior['I_init_mu']*1.2, prior['I_init_mu']*0.8)
    # I_init_mu = pm.Uniform('I_init_mu', -1, 1)

    # I_init = pm.Normal('I_init', mu=I_init_mu, sigma=I_init_std)

    
    # ae.mode='DebugMode'

    res, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
        'S': (S_initial, ()), 'I': (I_initial, ()), 'H': (H_initial, ())},  
        params={
        'beta': (beta, ()), 'lam': (lam, ()), 'gamma':(gamma, ()), 'delta': (delta, ()), '_dummy': (np.array(1.), ())},
        rhs=SIR_sunode,
        tvals=daterange.dayofyear,
        t0=daterange.dayofyear[0]
    )
    #if(likelihood['distribution'] == 'lognormal'):
    #    I = pm.Lognormal('I', mu=res['I'], sigma=sigma, observed=cases_obs_scaled)
    #elif(likelihood['distribution'] == 'normal'):

    I = pm.Normal('I', mu=res['I'], sigma=sigma, observed=cases_obs)
    H = pm.Normal('H', mu=res['H'], sigma=sigma_h, observed=hosp_admissions_obs)

    I_mu = pm.Deterministic('I_mu', res['I'])
    S_mu = pm.Deterministic('S_mu', res['S'])
    H_mu = pm.Deterministic('H_mu', res['H'])
    # R_mu = pm.Deterministic('R_mu', 1-S_mu-I)

    R0 = pm.Deterministic('R0',beta/lam)
    # step1 = pm.NUTS(target_accept=0.5, step_scale=0.1)    # still too slow
    # step1 = pm.NUTS(adapt_step_size=False, step_scale=0.2)
    # trace = pm.sample(n_samples, step=step1, chains=2, tune=n_tune, cores=8 
    trace = pm.sample(n_samples, chains=2, tune=n_tune, cores=8 
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
    #pm.plot_autocorr(trace)
    #plt.show()
    #pm.plot_trace(trace)
    #plt.show()
burned_trace = trace.isel(draw=slice(int(n_samples/4),-1))
burned_trace.to_netcdf("burned_trace_output_saved_on_disk_hosp5.nc")

"""
----------------------------------------------------------------------------------
Plotting & Saving
----------------------------------------------------------------------------------
"""
# Extract I, S, R values as an average at each timepoint from the burned trace and plot with the observed I.
Y = [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)]
arr = burned_trace.posterior.mean(dim="draw")
arr['I_mu'] = arr.I_mu.rename({'chain':'chain', 'I_mu_dim_0':'days_since_origin'})
arr['S_mu'] = arr.S_mu.rename({'chain':'chain', 'S_mu_dim_0':'days_since_origin'})
arr['H_mu'] = arr.H_mu.rename({'chain':'chain', 'H_mu_dim_0':'days_since_origin'})
Y[0] = arr['I_mu']
Y[1] = arr['S_mu']
Y[2] = arr['H_mu']
Y[3] = 1-Y[0]-Y[1]-Y[2]
# Yc = [  [ Y[y][c,:] for y in range(len(Y)) ] for c in range(2) ]

# now plot using dates as x, and I, S and R on the y-axis (Y[0], Y[1] and Y[2])
plt.plot(daterange, Y[0][0], "o--", label="Infected")
plt.plot(daterange, Y[1][0], "o--", label="Susceptible")
plt.plot(daterange, Y[2][0], "o--", label="Hospitalised")
plt.plot(daterange, Y[3][0], "o--", label="Removed")
plt.plot(daterange, cases_obs, "x--", label="I_obs")
plt.plot(daterange, hosp_admissions_obs, "x--", label="H_obs")
plt.ylim(0.000,0.1)
plt.legend(fontsize=12)

plt.show()

#Y[0].to_netcdf("I_output_saved_on_disk_run3.nc")

print('this is the end')