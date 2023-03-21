import pandas as pd


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    The wealth index
    The previous peaks
    Percent Drawdowns"""
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame(
        {
            "Wealth": wealth_index,
            "Previous Peaks": previous_peaks,
            "Drawdown": drawdowns,
        }
    )


def get_ffme_returns():
    me_m = pd.read_csv(
        "data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header=0,
        index_col=0,
        na_values=-99.99,
    )
    returns = me_m[["Lo 10", "Hi 10"]]
    returns.columns = ["SmallCap", "LargeCap"]
    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")
    return returns


def get_hfi_returns():
    """Load and format the EDHEC Hedge Fund Index Data"""
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv", header=0, index_col=0, 
        parse_dates=True
    )
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    return hfi

def semideviation(r):
    """
    Returns the semideviations aka Negative semideviation of r
    r must be a series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied series of DataFrame
    Returns a float or a Series
    """
    demeaned_r = r-r.mean()
    #use the population std, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied series of DataFrame
    Returns a float or a Series
    """
    demeaned_r = r-r.mean()
    #use the population std, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_nomal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value>level

import numpy as np

def var_historic (r, level=5):
    """
    Returns the historic value at risk at a scefified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number and the (100-level) percent are above
    """
    if isinstance (r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance (r, pd.Series):
        return -np.percentile(r, level)
    else:
        return TypeError("Expected r to be a Series or DataFrame")
    

from scipy.stats import norm
def var_gaussian(r, level=5, modified = False):
    """
    Returns the parametric Gaussian VaR of a Series or Dataframe
    """
    # compute the Z score assuming it was gaussian
    z = norm.ppf(level/100)
    if modified:
        # modified the Z score based on observed skewness and kurtosis
        s= skewness(r)
        k= kurtosis(r)
        z= (z+
            (z**2 - 1)*s/6+
            (z**3 - 3*z)*(k-3)/24-
            (2*z**3-5*z)*(s**2)/36
            )
    
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance (r, pd.Series):
        is_beyond = r <= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError ("Expected r to be a Series or DataFrame")

def get_ind_returns():
    ind = pd.read_csv('data\ind30_m_vw_rets.csv', header=0, index_col=0,parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def annualize_rets(r,periods_per_year):
    compound_growth = (1+r).prod()
    n_period = r.shape[0]
    return compound_growth**(periods_per_year/n_period)-1

def annualize_vola(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)

def sharp_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1  #convert the annual riskfree rate to per period
    excess_ret= r-rf_per_period
    ann_ex_ret = annualize_rets(excess_ret,periods_per_year)
    ann_vola =  annualize_vola(r, periods_per_year)
    return ann_ex_ret/ann_vola

def portfolio_return (weights, returns):
    """
    weights -> Return
    """
    return weights.T @ returns

def portfolio_vola (weights, covmat):
    """
    Weights -> Vola
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov):
    """ 
    Plot the 2 asset efficient frontier
    """
    if er.shape[0] !=2 or er.shape[0] !=2 :
        raise ValueError("plot_ef2 can only plot two asste")
    weights = [np.array([w,1-w])for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w,er)for w in weights]
    vols = [portfolio_vola(w,cov)for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")

from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
        """
        target_ret -> W
        """
        n= er.shape[0]
        init_guess = np.repeat(1/n,n)
        bounds = ((0.0,1.0),)*n
        return_is_target = {
            'type':'eq',
            'args':(er,),
            'fun': lambda weights, er : target_return - portfolio_return(weights,er)
        }
        weights_sum_to_1 = {
            'type':'eq',
            'fun': lambda weights: np.sum(weights)-1
        }
        results =minimize(portfolio_vola,init_guess,
                          args=(cov,), method='SLSQP',
                          options = {'disp': False},
                          constraints= (return_is_target, weights_sum_to_1),
                          bounds=bounds)
        return results.x  

def optimal_weights (n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharp ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n= er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharp_ratio (weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r= portfolio_return(weights, er)
        vol=portfolio_vola(weights, cov)
        return -(r-riskfree_rate)/vol
    results =minimize(neg_sharp_ratio,init_guess,
                          args=(riskfree_rate,er,cov,), method='SLSQP',
                          options = {'disp': False},
                          constraints= ( weights_sum_to_1),
                          bounds=bounds)
    return results.x

def gmv(cov):
    """
    Returns the weight of the Global Minimum Vol portfolio
    given the covariance matrix
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)  #By assuming a constant return of 1 for all assets in the portfolio, 
                                        # the focus is shifted entirely to the variance of the portfolio's returns, 
                                    # allowing for a simpler and more tractable optimization problem

def plot_ef (n_points, er, cov, show_cml=False, style='.-', riskfree_rate=0,show_ew=False, show_gmv=False):
    """
    Plots the multi-asset effecient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w,er)for w in weights]
    vols = [portfolio_vola(w,cov)for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })   
    ax=ef.plot.line(x="Volatility", y="Returns", style='.-')
    if show_ew:
        n= er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        vol_ew= portfolio_vola(w_ew,cov)
        #display EW
        ax.plot([vol_ew],[r_ew],color='goldenrod',marker="o",markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv= portfolio_vola(w_gmv,cov)
        #display GMV
        ax.plot([vol_gmv],[r_gmv],color='midnightblue',marker="o",markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vol_msr=portfolio_vola(w_msr,cov)
        #Add CML
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color='green',marker='o',linestyle="dashed",markersize=12,linewidth=2)
        return ax