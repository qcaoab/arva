import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/home/marcchen/Documents/factor_decumulation/researchcode/crsp code")


plots = True # set True for plots
if plots:
  plt.ion()

writeout = True # set true to write out final real index csv file
first_year = 1926
last_year = 2023 # update for future years (start of 2019 is end of 2018)

# Read in 10-year T-bond data from Homer
yrs, long_yield = np.loadtxt('homer.csv', delimiter=',', skiprows=1,
                             unpack=True)

# Construct monthly date series for interpolation of annual yields
obs_time_interp = (np.repeat(np.arange(yrs[0],yrs[-1]+1), 12) +
                  np.tile(np.arange(0,12)/12, yrs.size))
istart = np.where(obs_time_interp == yrs[0]+0.5)[0][0]
istop = np.where(obs_time_interp == yrs[-1]+0.5)[0][0]
obs_time_interp = obs_time_interp[istart:istop+1]

# Interpolation and plot (if desired)
y = np.interp(obs_time_interp, yrs+0.5, long_yield)
if plots:
    plt.figure()
    plt.plot(yrs+.5, long_yield, 'bo', markerfacecolor='none',
             label='Data from Homer')
    plt.plot(obs_time_interp, y, 'r', label='Interpolated')
    plt.title('Interpolated Long Yields')
    plt.legend(frameon=False)

# Index calculation from Homer data. Assume interpolated yields can be applied
# to a newly-issued 10-year bond paying semi-annual coupons and selling at par,
# so yield equals coupon rate. Each month, recalculate the price of the bond
# issued one month earlier given the prevailing yield. Sell that bond and
# invest the proceeds in the newly issued bond.
y /= 100 # Convert percentage to decimal
par = 100
coupons_per_year = 2
yp = y / coupons_per_year # yield per semi-annual period
original_maturity = 10 # years
bond_price = np.full(y.size, np.nan)
bond_price[0] = par
coupon_rate = y[0:-1]
pv_factor = (1 + yp[1:]) ** (-coupons_per_year * original_maturity)
annuity_factor = (1 - pv_factor) / yp[1:]
coupon = par*coupon_rate/coupons_per_year
bond_price[1:] = ((coupon * annuity_factor + par * pv_factor) *
                  (1 + yp[1:]) ** (coupons_per_year / 12))
bond_return = np.full(y.size, np.nan)
bond_return[1:] = bond_price[1:] / par - 1

# Delete observations prior to start of CRSP data in 1926
istart = np.where(obs_time_interp == first_year)[0][0]
obs_time_interp = obs_time_interp[istart:]
bond_return = bond_return[istart:]

# Import CRSP Treasury data and drop unneeded columns
dfT = pd.read_csv('treasury_2022.csv', index_col=0)
#df['caldt'] = pd.to_datetime(df['caldt'])
drop_cols = ['caldt', 'b30ret', 'b30ind', 'b20ret', 'b20ind', 'b7ret', 'b7ind',
             'b5ret', 'b5ind', 'b2ret', 'b2ind', 'b1ret', 'b1ind']
dfT.drop(columns=drop_cols, inplace=True)
Tdata = dfT.to_numpy()
b10ret = Tdata[:,0] # monthly return for 10-year T-bond
b10ind = Tdata[:,1] # total return index for 10-year T-bond
t90ret = Tdata[:,2] # monthly return for 90-day T-bill
t90ind = Tdata[:,3] # total return index for 90-day T-bill
t30ret = Tdata[:,4] # monthly return for 30-day T-bill
t30ind = Tdata[:,5] # total return index for 30-day T-bill
cpiret = Tdata[:,6] # monthly change in cpi
cpiind = Tdata[:,7] # cpi index

# Construct monthly date series for CRSP data
# Starting point is the beginning of 1926/end of 1925
# Last year is start of year following the last sample observation
obs_time_crsp = (np.repeat(np.arange(first_year, last_year), 12) +
                np.tile(np.arange(0,12)/12, last_year-first_year))
obs_time_crsp = np.append(obs_time_crsp, last_year)

# Overlapping period for Homer and CRSP Treasury data
Ncrsp = obs_time_crsp.size
Nhomer = obs_time_interp.size
bond_return_homer = np.full(Ncrsp, np.nan)
bond_return_homer[0:Nhomer] = bond_return 
start_overlap = np.where(np.isnan(b10ret))[0][-1] + 1
end_overlap = np.where(np.isnan(bond_return_homer))[0][0] # + 1
rh = bond_return_homer[start_overlap:end_overlap]
rc = b10ret[start_overlap:end_overlap]
xtime = obs_time_crsp[start_overlap:end_overlap]
if plots:
    plt.figure()
    plt.plot(xtime, rc, 'b', label='CRSP 10-Year Bond Return')
    plt.plot(xtime, rh, 'r', label='Homer Long Bond Return (Interpolated)')
    plt.title('Overlapping Data From Homer and CRSP')
    plt.legend(frameon=False)
    
# Construct merged Homer-CRSP 10-year bond index
mbr = np.full(Ncrsp, np.nan)
mbr[1:start_overlap] = 1 + bond_return[1:start_overlap]
mbr[start_overlap:] = 1 + b10ret[start_overlap:]
mbr[0] = 1.
mbi = 100. * np.cumprod(mbr)

# Rescale other bond indexes to 100 at start of 1926
b10indr = 100. * b10ind / b10ind[0]
t30indr = 100. * t30ind / t30ind[0]
t90indr = 100. * t90ind / t90ind[0]
cpiindr = 100. * cpiind / cpiind[0]

# Plot bond indexes and cpi
if plots:
    plt.figure()
    plt.plot(obs_time_crsp, mbi, 'b', label='Merged 10-Year T-Bond Index')
    plt.plot(obs_time_crsp, b10indr, 'r', label='CRSP 10-Year T-Bond Index')
    plt.plot(obs_time_crsp, t90indr, 'g', label='CRSP 90-Day T-Bill Index')
    plt.plot(obs_time_crsp, t30indr, 'c', label='CRSP 30-Day T-Bill Index')
    plt.plot(obs_time_crsp, cpiindr, 'k', label='CRSP CPI')
    plt.title('Bond Indexes')
    plt.legend(frameon=False)
    
# Import stock index data
dfS = pd.read_csv('stock_indexes_2022.csv', index_col=0)
drop_cols = ['date', 'sprtrn', 'spindx', 'totval',
             'totcnt', 'usdval', 'usdcnt']
dfS.drop(columns=drop_cols, inplace=True)
Sdata = dfS.to_numpy()
vwretd = Sdata[:,0] # Value-weighted return with distributions
vwretd[0] = 0
vwretx = Sdata[:,1] # Value-weighted return excluding distributions
vwretx[0] = 0
ewretd = Sdata[:,2] # Equal-weighted return with distributions
ewretd[0] = 0
ewretx = Sdata[:,3] # Equal-weighted return excluding distributions
ewretx[0] = 0
vwd = 100. * np.cumprod(1+vwretd) # Value-weighted index with distributions
vwx = 100. * np.cumprod(1+vwretx) # Value-weighted index excluding distributions
ewd = 100. * np.cumprod(1+ewretd) # Equal-weighted index with distributions
ewx = 100. * np.cumprod(1+ewretx) # Equal-weighted index excluding distributions

# Plot of nominal stock indexes and cpi
if plots:
   plt.figure()
   plt.plot(obs_time_crsp, vwd, 'b', label='CRSP Value-Weighted With Distributions')
   plt.plot(obs_time_crsp, vwx, 'c', label='CRSP Value-Weighted Without Distributions') 
   plt.plot(obs_time_crsp, ewd, 'r', label='CRSP Equal-Weighted With Distributions')
   plt.plot(obs_time_crsp, ewx, 'm', label='CRSP Equal-Weighted Without Distributions')
   plt.plot(obs_time_crsp, cpiindr, 'k', label='CRSP CPI')
   plt.title('Stock Indexes and CPI')
   plt.legend(frameon=False)
    
# Plot of all nominal indexes and cpi (log scale)
if plots:
    plt.figure()
    plt.semilogy(obs_time_crsp, t30indr, 'b--', label='CRSP 30-Day T-Bill')
    plt.semilogy(obs_time_crsp, t90indr, 'r--', label='CRSP 90-Day T-Bill')
    plt.semilogy(obs_time_crsp, mbi, 'c--', label='Merged 10-Year T-Bond')
    plt.semilogy(obs_time_crsp, vwd, 'g', label='CRSP Value-Weighted With Distributions')
    plt.semilogy(obs_time_crsp, vwx, 'b', label='CRSP Value-Weighted Without Distributions')
    plt.semilogy(obs_time_crsp, ewd, 'm', label='CRSP Equal-Weighted With Distributions')
    plt.semilogy(obs_time_crsp, ewx, 'r', label='CRSP Equal-Weighted Without Distributions')
    plt.semilogy(obs_time_crsp, cpiindr, 'k', label='CRSP CPI')
    plt.title('Nominal Indexes (Log Scale)')
    plt.legend(frameon=False)
    
# Real indexes (stored in matrix Y, with observation times and CPI)
Y = np.full((Ncrsp,9), np.nan)
Y[:,0] = obs_time_crsp
Y[:,1] = cpiindr
Y[:,2] = 100. * (t30indr/cpiindr)
Y[:,3] = 100. * (t90indr/cpiindr)
Y[:,4] = 100. * (mbi/cpiindr)
Y[:,5] = 100. * (vwd/cpiindr)
Y[:,6] = 100. * (vwx/cpiindr)
Y[:,7] = 100. * (ewd/cpiindr)
Y[:,8] = 100. * (ewx/cpiindr)

# Plot real indexes (regular scale)
if plots:
    plt.figure()
    plt.plot(Y[:,0], Y[:,2], 'b--', label='30-Day T-Bill')
    plt.plot(Y[:,0], Y[:,3], 'r--', label='90-Day T-Bill')
    plt.plot(Y[:,0], Y[:,4], 'c--', label='Merged 10-Year T-Bond')
    plt.plot(Y[:,0], Y[:,5], 'g', label='Value-Weighted With Distributions')
    plt.plot(Y[:,0], Y[:,6], 'b', label='Value-Weighted Without Distributions')
    plt.plot(Y[:,0], Y[:,7], 'm', label='Equal-Weighted With Distributions')
    plt.plot(Y[:,0], Y[:,8], 'r', label='Equal-Weighted Without Distributions')
    plt.title('Real Total Return Indexes')
    plt.legend(frameon=False)
    
# Plot real indexes (log scale)
if plots:
    plt.figure()
    plt.semilogy(Y[:,0], Y[:,2], 'b--', label='30-Day T-Bill')
    plt.semilogy(Y[:,0], Y[:,3], 'r--', label='90-Day T-Bill')
    plt.semilogy(Y[:,0], Y[:,4], 'c--', label='Merged 10-Year T-Bond')
    plt.semilogy(Y[:,0], Y[:,5], 'g', label='Value-Weighted With Distributions')
    plt.semilogy(Y[:,0], Y[:,6], 'b', label='Value-Weighted Without Distributions')
    plt.semilogy(Y[:,0], Y[:,7], 'm', label='Equal-Weighted With Distributions')
    plt.semilogy(Y[:,0], Y[:,8], 'r', label='Equal-Weighted Without Distributions')
    plt.title('Real Total Return Indexes (Log Scale)')
    plt.legend(frameon=False)
    
# Write to output file
if writeout:
    d = {'Time' : Y[:,0], 'CPI' : Y[:,1], 'T30' : Y[:,2], 'T90' : Y[:,3],
         'B10' : Y[:,4], 'VWD' : Y[:,5], 'VWX' : Y[:,6], 'EWD' : Y[:,7],
         'EWX' : Y[:,8]}
    dfw = pd.DataFrame(data=d)
    dfw.to_csv('real_indexes_2022.csv', index=False)

#------------------
# Nominal returns (stored in matrix X, with observation times and CPI)
X = np.full((Ncrsp,9), np.nan)
X[:,0] = obs_time_crsp
X[:,1] = 100 * cpiret
X[:,2] = 100 * (t30ret)
X[:,3] = 100 * (t90ret)
X[:,4] = 100 * (mbr) - 100
X[:,5] = 100 * (vwretd)
X[:,6] = 100 * (vwretx)
X[:,7] = 100 * (ewretd)
X[:,8] = 100 * (ewretx)

    
# Write to output file
if writeout:
    d = {'Time' : X[:,0], 'CPI' : X[:,1], 'T30' : X[:,2], 'T90' : X[:,3],
         'B10' : X[:,4], 'VWD' : X[:,5], 'VWX' : X[:,6], 'EWD' : X[:,7],
         'EWX' : X[:,8]}
    dfw = pd.DataFrame(data=d)
    dfw.to_csv('nom_returns_2022.csv', index=False)
