# T = 30
# B: Real_30_day_Tbill_1926_2017.dat
# S: Real_CRSP_1926_2017.dat
# W0: 1000
# cash withdrawal: 0, 1, ..., 30
# Q max limit: 100
# rebalancing interval: 1 year
#---------------------------------
#data generation
#number of generated paths: 640000
# market parameters
#---------------------------------
# CRSP
# mu: 0.08753
# sigma: 0.14801
# epsilon: 0.34065
# pu: 0.25806
# ita1: 4.67877
# ita2: 5.60389
#----------------------------------
# Treasury Bill
# mean return: 0.004835
# volatility: 0.018920
# correlation: 0.06662

'''params["T"] = 30. #Investment time horizon, in years
params["W0"] = 1000.     # Initial wealth W0
params["N_rb"] = 30  #Nr of equally-spaced rebalancing events in [0,T]
params["delta_t"] = params["T"] / params["N_rb"]    # Rebalancing time interval 1 year
params["N_d_train"] = 6400000 #int(2.56* (10**6)) #Nr of TRAINING data return sample paths to bootstrap'''
Qmax = 100.

Annuity = [0.038441698899,
 0.039737446742,
 0.041125877396,
 0.042616541642,
 0.044220375891,
 0.045949611166,
 0.047818260291,
 0.049842262216,
 0.052040178528,
 0.05443317918,
 0.057046076772,
 0.059907921525,
 0.063052415788,
 0.066497653234,
 0.070266432327,
 0.074437371772,
 0.079070689514,
 0.084240205882,
 0.090037552518,
 0.096452912391,
 0.10350960148,
 0.11152017659,
 0.12071687619,
 0.13061738207,
 0.1419146925,
 0.15516289276,
 0.16884516228,
 0.18573778221,
 0.20249470457,
 0.22372590524,
 0.24310095479]


