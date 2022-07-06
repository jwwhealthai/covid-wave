from datetime import datetime
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
def COVID_pol_data():
    try:
        raw_OxCGRT = pd.read_csv("data/policy/OxCGRT_latest.csv")
    except:
        "Policy data is not loading. Check that it's a valid csv file."
    raw_data_p =raw_OxCGRT.loc[(raw_OxCGRT['CountryCode']=='GBR')&(raw_OxCGRT['Jurisdiction']=='NAT_TOTAL')][['Date','StringencyIndex']]
    raw_data_p['date'] = pd.to_datetime(raw_data_p['Date'], format="%Y%m%d")
    policy_ind = raw_data_p.dropna().reset_index().dropna().drop(columns='index')
    return policy_ind

def COVID_case_data():
    try:
        raw_data_c = pd.read_csv("data/cases/data_2021-Jul-13.csv") # 'newCasesByPublishDate'
    except:
        "Covid case incidence data is not loading. Check that it's a valid csv file."
    raw_data_c['date_new'] = pd.to_datetime(raw_data_c['date'])         # coerces date column to datetime
    raw_data_c = raw_data_c.sort_values(by='date_new' , axis='index')   # sorts so that earliest date appears first
    cases = raw_data_c[['newCasesByPublishDate','date_new']]          # makes dataframe of date and cases columns
    cases = cases.reset_index()
    return cases

def COVID_hosp_data():
    try:
        raw_data_h = pd.read_csv("data/hosp/data_2021-Jun-17.csv") # 'newAdmissions'
    except:
        "Covid hospitalisation incidence data is not loading. Check that it's a valid csv file."
    raw_data_h['date_new'] = pd.to_datetime(raw_data_h['date'])         # coerces date column to datetime
    raw_data_h = raw_data_h.sort_values(by='date_new' , axis='index')   # sorts so that earliest date appears first
    tmp_data_h = raw_data_h.groupby(by=[raw_data_h.date_new], as_index=False).sum()     # groups 4 nations data by date and sums admissions
    assert tmp_data_h.shape[0] < raw_data_h.shape[0], "Raw hospitalisation dataset is the same length or longer than it is after grouping all 4 nations by date and aggregating. Something must be wrong with the input data or grouping code."
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
    I_initial_disagg = covid_case_obj.iloc[(covid_case_obj.index[covid_case_obj.date_new==daterange[0]].values[0] - infectious_duration(tau)) : covid_case_obj.index[covid_case_obj.date_new==daterange[0]].values[0]+1]['newCasesByPublishDate']
    assert I_initial_disagg.shape[0] == tau+1, "I_initial not considering the right number of days of infections"
    I_initial = np.sum(I_initial_disagg)
    # I0 = pd.Series(I_initial)
    # I0.name = 'I0'
    # I0 = covid_case_obj.loc[(covid_case_obj['date_new'] > starttime - dt.timedelta(days=tau)) & (covid_case_obj['date_new'] <= starttime)]['newCasesByPublishDate']
    return I_initial

def get_initial_H(tau, covid_hosp_obj, daterange):
    H_initial_disagg = covid_hosp_obj.iloc[(covid_hosp_obj.index[covid_hosp_obj.date_new==daterange[0]].values[0] - infectious_duration(tau)) : covid_hosp_obj.index[covid_hosp_obj.date_new==daterange[0]].values[0]+1]['newAdmissions']
    assert H_initial_disagg.shape[0] == tau+1, "H_initial not considering the right number of days of hospitalisations"
    H_initial = np.sum(H_initial_disagg)
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
    outputs, __ = ae.scan(fn=increment_t, sequences=[beta], outputs_info=[S_t_init, I_t_init, H_t_init, I_new_init, H_new_init], non_sequences=[lam, gamma, delta, N])
    # final_outputs = outputs[-1]

    # N = aet.iscalar()
    # check = ae.function(inputs=[beta, lam, gamma, delta, S_t_init, I_t_init, H_t_init], outputs = final_outputs, updates=updates, mode='DebugMode')
    # print(check(1.3, 0.09, 0.05, 0.08, 66998000, 1305, 200))


    S_t_all, I_t_all, H_t_all, I_new_all, H_new_all = outputs
    return S_t_all, I_t_all, H_t_all, I_new_all, H_new_all
         

# Test model:

# beta = aet.dmatrix('beta')
# lam = aet.dscalar('lam')
# gamma = aet.dscalar('gamma')
# delta = aet.dscalar('delta')
# S_t_init = aet.lscalar('S_t_init')
# I_t_init = aet.lscalar('I_t_init')
# H_t_init = aet.lscalar('H_t_init')
# N = aet.lscalar('N')
# S, I, H, I_new, H_new = SIHR(beta = 1 * aet.ones(5), lam=1, gamma=1, delta=1, S_t_init=100, I_t_init=aet.dscalar(10), H_t_init=0, N=1000)

"""
---------------------------------------------------------------------------------
Configuration settings, parameter priors etc.
---------------------------------------------------------------------------------
"""
start = "2020-03-15"
end = "2020-04-30"
daterange = pd.date_range(start=start, end=end)
ndays = len(daterange.day)
covid_case_obj = COVID_case_data()                      # Make a function to extract case data when given a country code argument
covid_hosp_obj = COVID_hosp_data()
covid_poli_obj = COVID_pol_data()

# Some tests to ensure data integrity during dev
first_date_cases = covid_case_obj['date_new'][0]
last_date_cases = covid_case_obj['date_new'].iloc[-1]
first_date_hosp = covid_hosp_obj['date_new'][0]
last_date_hosp = covid_hosp_obj['date_new'].iloc[-1]
first_date_policy = covid_poli_obj['date'][0]
last_date_policy = covid_poli_obj['date'].iloc[-1]

assert (first_date_cases < last_date_cases), 'first date in index is not earlier than last, there must be a sorting or indexing issue.'
assert (first_date_hosp < last_date_hosp), 'first date in index is not earlier than last, there must be a sorting or indexing issue.'
assert (first_date_policy < last_date_policy), 'first date in index is not earlier than last, there must be a sorting or indexing issue.'
assert len(pd.date_range(first_date_policy, last_date_policy)) == covid_poli_obj.shape[0], 'daterange of policy date is not equal to its number of rows, suggesting there are missing dates or a grouping issue (ought to be 1 row per daily date)'

n_samples = 1000
n_tune = 200
N = 67000000                                                     # Population of UK
tau = 10                                                         # Num days people remain in infectious compartment, used for rough estimate of I_initial from cases[0]
I_initial = get_initial_I(tau, covid_case_obj, daterange)     # From observed data, takes the case numbers of the tau days prior to startdate.
H_initial = get_initial_H(tau, covid_hosp_obj, daterange)     # From observed data, takes the admissions numbers of the tau days prior to startdate.
# Prior estimate of R_initial sums all cases up to 10 days before start date.
R_initial = np.sum(covid_case_obj.loc[covid_case_obj['date_new']<= daterange[0]-tau*daterange.freq]['newCasesByPublishDate'])
S_initial = (N - R_initial - I_initial - H_initial)   # All cases removed before startdate-tau incl in R_initial. Infected or hospitalised cases within startdate-tau also need to subtracted.
cases_obs = covid_case_obj.loc[(covid_case_obj['date_new'] >= daterange[0]) & (covid_case_obj['date_new'] < daterange[-1])]['newCasesByPublishDate']
hosp_admissions_obs = covid_hosp_obj.loc[(covid_hosp_obj['date_new'] >= daterange[0]) & (covid_hosp_obj['date_new'] < daterange[-1])]['newAdmissions']
policy_obs = covid_poli_obj.loc[(covid_poli_obj['date'] >= daterange[0]) & (covid_poli_obj['date'] < daterange[-1])]['StringencyIndex']

def calc_logn_params(mu_est, sd_est):
    mu = np.log( mu_est**2 / np.sqrt(mu_est**2 + sd_est**2) )
    sig = np.sqrt(np.log(1 + ( sd_est**2 / mu_est**2 )))
    return dict(mu = mu, sig = sig)

prior = {   'beta_int': calc_logn_params(0.5, 0.03),
            'beta_grad': calc_logn_params(0.0035, 0.0002),
            # 'scale_mu': np.log(0.02),
            # 'scale_sig': 0.02,
            # 'beta_pol_sig': 0.3,
            # 'beta_inf_mu': np.log(0.5),
            # 'beta_inf_sig': (0.25),
            'gamma': calc_logn_params(0.05, 0.003),
            'delta': calc_logn_params(0.25, 0.09),
            'lam': calc_logn_params(0.15, 0.06),
            'S_t_init': calc_logn_params(S_initial-10000, S_initial*10**-5),
            'I_t_init': calc_logn_params(I_initial, 50),
            'H_t_init': calc_logn_params(H_initial, 25)
            # 'I_obs_err_mu': 1.5,
            # 'I_obs_err_sig':0.9
        }


"""
----------------------------------------------------------------------------------
Perform SIHR modelling on waves 1 and 2
----------------------------------------------------------------------------------
"""
with pm.Model() as model:

    # I_t_init = pm.Lognormal("I_t_init", prior['I_t_init_mu'], prior['I_t_init_sig'], initval=I_initial)
    # H_t_init = pm.Lognormal("H_t_init", prior['H_t_init_mu'], prior['H_t_init_sig'], initval=H_initial)
    # #R_t_init = pm.Normal("R_t_init", R_initial, R_initial*0.01)
    # #S_t_init_mu = pm.Deterministic("S_t_init_mu", 1-I_t_init-H_t_init-R_t_init)
    # S_t_init = pm.Lognormal("S_t_init", prior['S_t_init_mu'], prior['S_t_init_sig'], initval=S_initial)    # N is equal to 1, R will be > 0 if the daterange starts after the beginning of first wave

    I_t_init = pm.Lognormal("I_t_init", prior['I_t_init']['mu'], prior['I_t_init']['sig'])
    H_t_init = pm.Lognormal("H_t_init", prior['H_t_init']['mu'], prior['H_t_init']['sig'])
    #R_t_init = pm.Normal("R_t_init", R_initial, R_initial*0.01)
    #S_t_init_mu = pm.Deterministic("S_t_init_mu", 1-I_t_init-H_t_init-R_t_init)
    S_t_init = pm.Lognormal("S_t_init", prior['S_t_init']['mu'], prior['S_t_init']['sig'])    # N is equal to 1, R will be > 0 if the daterange starts after the beginning of first wave

    beta_int = pm.Lognormal('beta_int', prior['beta_int']['mu'], prior['beta_int']['sig'])
    beta_grad = pm.Lognormal('beta_grad', prior['beta_grad']['mu'], prior['beta_grad']['sig'])
    beta = pm.Deterministic('beta', beta_int - (beta_grad * policy_obs.to_numpy()))
    # beta_pol = pm.Lognormal('beta_pol', mu=np.log(policy_obs).to_numpy(), sigma=prior['beta_pol_sig'])
    # scale = pm.Lognormal('scale', mu=prior['scale_mu'], sigma=prior['scale_sig'])
    # policy_factor = pm.Deterministic('policy_factor', beta_pol*scale)
    # beta_inf = pm.Lognormal('beta_inf', prior['beta_inf_mu'], prior['beta_inf_sig'])
    # beta = pm.Deterministic('beta', beta_inf/policy_factor)         # A low stringency level (i.e. below 1) will increase infection rate, whereas stringency above 1 will reduce it. 
    lam = pm.Lognormal('lam', prior['lam']['mu'], prior['lam']['sig'])
    gamma = pm.Lognormal('gamma', prior['gamma']['mu'], prior['gamma']['sig'])
    delta = pm.Lognormal('delta', prior['delta']['mu'], prior['delta']['sig'])
    # I_obs_err = pm.Lognormal('I_obs_err', prior['I_obs_err_mu'], prior['I_obs_err_sig'])
    # case_obs_err = pm.HalfCauchy('case_obs_err', beta=0.00005)    # RV for error in case collection figures
    # adm_obs_err = pm.HalfCauchy('adm_obs_err', beta=0.000005)      # RV for error in hospital admissions figures
    S, I, H, I_new, H_new = SIHR(beta=beta * aet.ones(ndays-1), lam=lam, 
                                               gamma=gamma, delta=delta, S_t_init=S_t_init, I_t_init=I_t_init, 
                                               H_t_init=H_t_init, N=N)
    # re:student ttest: nu should be roughly equivalent to the (number of samples) / (number of days in the daterange) - 1
    new_cases = pm.StudentT('new_cases', nu=4, mu=I_new, sigma=4000, observed=cases_obs)
    new_admissions = pm.StudentT('new_admissions', nu=4, mu=H_new, sigma=3, observed=hosp_admissions_obs)

    S = pm.Deterministic('S', S)
    I = pm.Deterministic('I', I)
    H = pm.Deterministic('H', H)
    R = pm.Deterministic('R', N-S-I-H)
    I_new = pm.Deterministic('I_new', I_new)
    H_new = pm.Deterministic('H_new', H_new)

    R0 = pm.Deterministic('R0',beta/lam)

    # Checks on priors (run first 4 lines within model, next 3 from debug console)
    RANDOM_SEED = 8157
    # np.random.seed(286)
    prior_checks = pm.sample_prior_predictive(samples=50, random_seed=RANDOM_SEED)
    # interdata_prior = az.from_pymc3(prior=prior_checks)
    
    # _, ax = plt.subplots()
    # interdata_prior.prior.plot.scatter(x="new_admissions", y="adm_obs_err", ax=ax)
    # # plt.show()

    # Sampling
    # step = pm.Metropolis()
    step = pm.NUTS()    
    # step1 = pm.NUTS(adapt_step_size=False)
    # trace = pm.sample(n_samples, step=step1, chains=2, tune=n_tune, cores=8 
    trace = pm.sample(draws=n_samples, step=step, init='adapt_diag', tune=n_tune, chains=2, cores=1)

trace.to_netcdf("raw_discretised_29.nc")
burned_trace = trace.isel(draw=slice(int(n_samples/4),-1))
burned_trace.to_netcdf("discretised_29.nc")
final_trace = burned_trace.isel(draw=slice(int(n_samples*.99),-1))

"""
----------------------------------------------------------------------------------
Plotting & Saving
----------------------------------------------------------------------------------
"""
# Extract I, S, R values as an average at each timepoint from the burned trace and plot with the observed I.
arr = burned_trace.posterior.mean(dim='draw')
arr_obs = trace.observed_data
# arr = burned_trace_post.mean(dim='draw')
# arr = burned_trace_post.mean(dim='draw')
# arr = raw_trace_post.isel(draw=slice(-2, -1))
# arr = arr.mean(dim='draw')
# arr_obs = trace_obs
Y = [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)]
arr['I_new'] = arr.I_new.rename({'chain':'chain', 'I_new_dim_0':'days_since_origin'})
arr['S'] = arr.S.rename({'chain':'chain', 'S_dim_0':'days_since_origin'})
arr['H_new'] = arr.H_new.rename({'chain':'chain', 'H_new_dim_0':'days_since_origin'})
arr['R'] = arr.R.rename({'chain':'chain', 'R_dim_0':'days_since_origin'})
arr['I'] = arr.I.rename({'chain':'chain', 'I_dim_0':'days_since_origin'})
arr['H'] = arr.H.rename({'chain':'chain', 'H_dim_0':'days_since_origin'})
# Y[0] = arr['new_cases']
Y[0] = arr['I_new']
Y[2] = arr['S']
# Y[2] = arr['new_admissions']
Y[1] = arr['H_new']
Y[3] = arr['R']
Y[4] = arr['I']
Y[5] = arr['H']
# daterange = pd.date_range(start='2020-05-09', end='2020-06-01', freq='D')
max_x_range = len(daterange)-1
x_ax_range = np.linspace(1,max_x_range, max_x_range)
# now plot using dates as x, and I, S and R on the y-axis (Y[0], Y[1] and Y[2])
plt.plot(x_ax_range, Y[0][1], "o--", label="new infections1")
plt.plot(x_ax_range, Y[1][1], "o--", label="new admissions1")
plt.plot(x_ax_range, Y[2][1], "o--", label="Susceptible")
plt.plot(x_ax_range, Y[3][1], "o--", label="Removed")
plt.plot(x_ax_range, Y[4][1], "o--", label="Current infection level")
plt.plot(x_ax_range, Y[5][1], "o--", label="Current hospitalised level")

plt.plot(x_ax_range, arr_obs['new_cases'], "x--", label="I_obs_new")
plt.plot(x_ax_range, arr_obs['new_admissions'], "x--", label="H_obs_new")
plt.ylim(0,11000)
plt.legend(fontsize=8)

plt.show()


print('this is the end')