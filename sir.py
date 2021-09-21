import aesara as ae
import aesara.tensor as aet
import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import xarray as xa
from netCDF4 import Dataset

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

"""
---------------------------------------------------------------------------------
Daily difference system which defines the model (SIHR)
---------------------------------------------------------------------------------
"""
# Discretising the ODE model might yield significant speed-ups. Therefore make delta_t equal to 1 (i.e. time interval = 1 day).
# This function will determine values for S, I and H given: beta, lambda, gamma and delta arguments (random variable priors) 
# as well as initial S (a random variable prior), I (a RV prior), H (RV prior) and N (UK population, fixed).

def SIHR(beta, lam, gamma, delta, S_t_init, I_t_init, H_t_init, N):
    I_new_init = aet.zeros_like(I_t_init)
    H_new_init = aet.zeros_like(H_t_init)
    def increment_t(beta, S_prev_t, I_prev_t, H_prev_t, temp0, temp1, lam, gamma, delta, N):
        I_new = beta/N*S_prev_t*I_prev_t                  # all those with 'new' subscript only consider in-flows
        H_new = gamma*I_prev_t
        I_t = I_new -lam*I_prev_t - H_new + I_prev_t
        S_t = S_prev_t - I_new
        H_t = H_new + H_prev_t - delta*H_prev_t
        return S_t, I_t, H_t, I_new, H_new
    # sequences is the 'list' that the theano loop iterates over similar to how python loops over an array, len(array) times. Beta is a tensor of ndays elements with each taking the value of the current beta RV equiv.
    # outputs_info accumulates results. It indicates to 'scan' that the results from the prior iteration (or the initial values) for these variables need to be passed to increment_t.
    # non-sequences such as lam, gamma, delta are automatically detected by scan because they are used within increment_t, although performance might increase if they're explicitly declared
    outputs, _ = ae.scan(fn=increment_t, sequences=[beta], outputs_info=[S_t_init, I_t_init, H_t_init, I_new_init, H_new_init], non_sequences=[lam, gamma, delta, N])
    S_t_all, I_t_all, H_t_all, I_new_all, H_new_all = outputs
    return S_t_all, I_t_all, H_t_all, I_new_all, H_new_all
         

"""
---------------------------------------------------------------------------------
Configuration settings, parameter priors etc.
---------------------------------------------------------------------------------
"""
daterange = pd.date_range(start="2020-03-15", end="2020-04-02")
ndays = len(daterange.day)
covid_case_obj = COVID_case_data(daterange)                      # Make a function to extract case data when given a country code argument
covid_hosp_obj = COVID_hosp_data(daterange)

n_samples = 400
n_tune = 1000
N = 67000000                                                     # Population of UK
tau = 10                                                         # Num days people remain in infectious compartment, used for rough estimate of I_initial from cases[0]
I_initial = get_initial_I(tau, covid_case_obj, daterange)     # From observed data, takes the case numbers of the tau days prior to startdate.
H_initial = get_initial_H(tau, covid_hosp_obj, daterange)     # From observed data, takes the admissions numbers of the tau days prior to startdate.
# Prior estimate of R_initial sums all cases up to 10 days before start date.
R_initial = np.sum(covid_case_obj.loc[covid_case_obj['date_new']<= daterange[0]-tau*daterange.freq]['newCasesByPublishDate'])
S_initial = (N - R_initial - I_initial - H_initial)   # All cases removed before startdate-tau incl in R_initial. Infected or hospitalised cases within startdate-tau also need to subtracted.
cases_obs = covid_case_obj.loc[(covid_case_obj['date_new'] >= daterange[0]) & (covid_case_obj['date_new'] < daterange[-1])]['newCasesByPublishDate']
hosp_admissions_obs = covid_hosp_obj.loc[(covid_hosp_obj['date_new'] >= daterange[0]) & (covid_hosp_obj['date_new'] < daterange[-1])]['newAdmissions']

prior = {   'beta_mu': np.log(2/5),
            'beta_sig': (0.5),
            'gamma_mu': np.log(1/10),
            'gamma_sig': (0.5),
            'delta_mu': np.log(1/12),
            'delta_sig': (0.4),
            'lam_mu': np.log(1/8),
            'lam_sig': (0.5),
            'S_t_init_mu': np.log(S_initial),
            'S_t_init_sig': 0.03,
            'I_t_init_mu': np.log(I_initial),
            'I_t_init_sig': 0.9,
            'H_t_init_mu': np.log(H_initial),
            'H_t_init_sig': 0.3
        }


"""
----------------------------------------------------------------------------------
Perform SIHR modelling on waves 1 and 2
----------------------------------------------------------------------------------
"""
with pm.Model() as model:

    I_t_init = pm.LogNormal("I_t_init", prior['I_t_init_mu'], prior['I_t_init_sig'])
    H_t_init = pm.LogNormal("H_t_init", prior['H_t_init_mu'], prior['H_t_init_sig'])
    #R_t_init = pm.Normal("R_t_init", R_initial, R_initial*0.01)
    #S_t_init_mu = pm.Deterministic("S_t_init_mu", 1-I_t_init-H_t_init-R_t_init)
    S_t_init = pm.LogNormal("S_t_init", prior['S_t_init_mu'], prior['S_t_init_sig'])    # N is equal to 1, R will be > 0 if the daterange starts after the beginning of first wave

    # sigma = pm.HalfCauchy('sigma', likelihood['sigma'])
    # sigma_h = pm.HalfCauchy('sigma_h', likelihood['sigma_h'])
    # H_sigma = pm.HalfCauchy('H_sigma', likelihood['H_sigma'])
    beta = pm.LogNormal('beta', prior['beta_mu'], prior['beta_sig'])     # lognormal might not be appropriate
    lam = pm.LogNormal('lambda', prior['lam_mu'], prior['lam_sig'])
    gamma = pm.LogNormal('gamma', prior['gamma_mu'], prior['gamma_sig'])
    delta = pm.LogNormal('delta', prior['delta_mu'], prior['delta_sig'])
    # case_obs_err = pm.HalfCauchy('case_obs_err', beta=0.00005)    # RV for error in case collection figures
    # adm_obs_err = pm.HalfCauchy('adm_obs_err', beta=0.000005)      # RV for error in hospital admissions figures

    S, I, H, I_new, H_new = SIHR(beta=beta * aet.ones(ndays-1), lam=lam, 
                                               gamma=gamma, delta=delta, S_t_init=S_t_init, I_t_init=I_t_init, 
                                               H_t_init=H_t_init, N=N)


# re:student ttest: nu should be roughly equivalent to the (number of samples) / (number of days in the daterange) - 1
    new_cases = pm.StudentT('new_cases', nu=4, mu=I_new, sigma=I_new, observed=cases_obs)
    new_admissions = pm.StudentT('new_admissions', nu=4, mu=H_new, sigma=1, observed=hosp_admissions_obs)

    S = pm.Deterministic('S', S)
    I = pm.Deterministic('I', I)
    H = pm.Deterministic('H', H)
    R = pm.Deterministic('R', N-S-I-H)
    I_new = pm.Deterministic('I_new', I_new)
    H_new = pm.Deterministic('H_new', H_new)

    R0 = pm.Deterministic('R0',beta/lam)

    # Checks on priors (run first 4 lines within model, next 3 from debug console)
    # RANDOM_SEED = 8157
    # np.random.seed(286)
    # prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)      # takes a minute or so to run
    # interdata_prior = az.from_pymc3(prior=prior_checks)
    
    # # _, ax = plt.subplots()
    # # interdata_prior.prior.plot.scatter(x="new_admissions", y="adm_obs_err", ax=ax)
    # # plt.show()

    # Sampling
    # step = pm.Metropolis()
    # step = pm.NUTS()    # still too slow
    # step1 = pm.NUTS(adapt_step_size=False)
    # trace = pm.sample(n_samples, step=step1, chains=2, tune=n_tune, cores=8 
    trace = pm.sample(draws=n_samples, tune=n_tune, chains=2, cores=8)

trace.to_netcdf("raw_discretised_10.nc")
burned_trace = trace.isel(draw=slice(int(n_samples/4),-1))
burned_trace.to_netcdf("discretised_10.nc")
final_trace = burned_trace.isel(draw=slice(int(n_samples*.99),-1))

"""
----------------------------------------------------------------------------------
Plotting & Saving
----------------------------------------------------------------------------------
"""
# Extract I, S, R values as an average at each timepoint from the burned trace and plot with the observed I.
# arr = burned_trace.posterior.mean(dim='draw')
# arr_obs = trace.observed_data
# arr = burned_trace_post.mean(dim='draw')
arr = burned_trace_post.mean(dim='draw')
# arr = raw_trace_post.isel(draw=slice(-2, -1))
# arr = arr.mean(dim='draw')
arr_obs = trace_obs
Y = [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)]
arr['I_new'] = arr.I_new.rename({'chain':'chain', 'I_new_dim_0':'days_since_origin'})
arr['S'] = arr.S.rename({'chain':'chain', 'S_dim_0':'days_since_origin'})
arr['H_new'] = arr.H_new.rename({'chain':'chain', 'H_new_dim_0':'days_since_origin'})
arr['R'] = arr.R.rename({'chain':'chain', 'R_dim_0':'days_since_origin'})
# Y[0] = arr['new_cases']
Y[0] = arr['I_new']
Y[2] = arr['S']
# Y[2] = arr['new_admissions']
Y[1] = arr['H_new']
Y[3] = arr['R']
# daterange = pd.date_range(start='2020-05-09', end='2020-06-01', freq='D')
max_x_range = len(daterange)-1
x_ax_range = np.linspace(1,max_x_range, max_x_range)
# now plot using dates as x, and I, S and R on the y-axis (Y[0], Y[1] and Y[2])
plt.plot(x_ax_range, Y[0][0], "o--", label="new infections")
plt.plot(x_ax_range, Y[1][0], "o--", label="new admissions")
plt.plot(x_ax_range, Y[2][0], "o--", label="Susceptible")
plt.plot(x_ax_range, Y[3][0], "o--", label="Removed")
plt.plot(x_ax_range, arr_obs['new_cases'], "x--", label="I_obs_new")
plt.plot(x_ax_range, arr_obs['new_admissions'], "x--", label="H_obs_new")
# plt.ylim(0.000,0.001)
plt.legend(fontsize=12)

plt.show()


print('this is the end')