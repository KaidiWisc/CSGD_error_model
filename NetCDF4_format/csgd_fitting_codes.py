# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:11:19 2022

@author: Kaidi Peng

CSGD error model
additional functions

Reference:

    Wright, D. B., Kirschbaum, D. B., & Yatheendradas, S. (2017). 
    Satellite Precipitation Characterization, Error Modeling, 
    and Error Correction Using Censored Shifted Gamma Distributions. 
    Journal of Hydrometeorology, 18(10), 2801-2815. 
    https://doi.org/10.1175/JHM-D-17-0060.1

"""

#%% (1) import package
import numpy as np
import scipy as sp
from scipy.special import beta

import math
import sys

#%% (2) CSGD-related Probabilistic Functions

# Generate N random variates of the CSGD
def GenerateRandEnsemble(CondParams,nensembles):
    qspace=np.random.uniform(0.,1.,nensembles)
    qcsgd2(qspace,CondParams)
    return np.apply_along_axis(qcsgd2, 0,qspace,CondParams) 


# Generate specific quantiles of the CSGD
# e.g. quantiles array [0.05,0.5,0.95] for 5th, 50th & 95th percentiles;
def GenerateSpecQuant(CondParams,quantiles):
    qspace=np.array(quantiles)
    qcsgd2(qspace,CondParams)
    return np.apply_along_axis(qcsgd2, 0,qspace,CondParams) 


# Return CSGD cumulative distribution function CDF at value x
def pcsgd(x,pars):
    # mu, sigma, shift
    mu=pars[0]
    sigma=pars[1]
    shift=pars[2]
    return sp.stats.gamma.cdf(x-shift,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)

        
# Return CSGD probability density function PDF at value x
def dcsgd(x,pars):
    # mu, sigma, shift
    mu=pars[0]
    sigma=pars[1]
    shift=pars[2]
    return sp.stats.gamma.pdf(x-shift,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)

  
# Return CSGD percent point function PPF (the inverse of CDF) at probability p
def qcsgd(p,pars):
    # mu, sigma, shift
    mu=pars[0]
    sigma=pars[1]
    shift=pars[2]
    quants=shift+sp.stats.gamma.ppf(p,((mu/sigma)**2),scale=(sigma**2)/mu,loc=0)
    # quants[quants<0.]=0.
    if quants<0:
        quants=0

    
    return quants

def qcsgd2(p,pars):
    # mu,sigma,shift
    mu=pars[0]
    sigma=pars[1]
    shift=pars[2]
    quants=shift+sp.stats.gamma.ppf(p,np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=0)
        
    quants[quants<0.]=0.
    
    return quants

 

# Return empirical cumulative distribution function of input data
def modecdf(data,npt):
    
    # probability of no precipitation
    pnp=len(data[data<=npt])/len(data)
    
    x = np.sort(data)   
    n = x.size
    y = np.arange(1, n+1) / n
    
    x0 = x[x>npt]
    y0 = y[x>npt]
       
    xp = np.insert(x0,0,npt)
    yp = np.insert(y0,0,pnp)
    
    return(xp,yp)


#%% (3) Optimization Functions of Climatological CSGD Error Model

# 3.1 Constrained and Unconstrained Optimization Functions
def fitcsgd_clim(x, bnds=False, initguess=False, constrained=False): 
    
    # Bounds for Clim. CSGD PARS: [mu, sigma, delta]
    if bnds==False:
        bnds = ((0.00001, None), (0.00001, None), (None,-0.0001)) 

    
    x = x[np.isfinite(x)].astype('float64')
    pnorain = np.float64(np.sum(np.equal(x,0.)))/len(x)
    pop = 1.-pnorain  
    
    # Initial Values for optimization
    if len(x)<50:
        mu_clim = np.nan
        sigma_clim = np.nan
        delta_clim = np.nan
    
    else:
        startmu = np.nanmean(x)
        startsigma = startmu
        startdelta = startmu*np.log(pop)
              
        if np.all(initguess==False):
            # Data-driven Initial Guesses (default)
            pst=[startmu, startsigma, startdelta]
            
        else:
            # Arbitrary Initial Guesses
            pst=[2.54650, 6.60258, -0.25606] 
        
        
        # (1) Constrained Scenario: Shift Constraint: delta>=-mu; 
        # constrained==True is recommended
        if constrained==True:
            
            def shiftcons(p):
                return p[0]+p[2]  # mu+delta >=0
        
            cons = ({'type': 'ineq', 'fun': shiftcons})
            csgdfit=sp.optimize.minimize(FitShiftedGamma, pst, args=(x,), method='SLSQP', bounds=bnds, 
                                          options={'maxiter': 10000, 'disp': False, 'ftol': 1e-10}, constraints=cons)
                      
        
        # (2) Unconstrained Scenario: 
        if constrained==False:
            csgdfit=sp.optimize.minimize(FitShiftedGamma, pst, args=(x,), method='L-BFGS-B', 
                                      bounds=bnds, options={'disp': False, 'ftol':1e-15})
        

    
    return csgdfit, pst


# 3.2 Perform the CRPS minimization to estimate clim. CSGD parameters
def FitShiftedGamma(p,x):
    
    k=p[0]*p[0]/p[1]/p[1]      # shape 
    theta=p[1]*p[1]/p[0]       # scale
    delta=p[2]                 # shift
    
    
    y=x[np.isfinite(x)].astype('float64')
    crps=np.empty((len(y)),dtype='float64')
    
    
    betaf=beta(0.5,0.5+k)
    ysq=(y[np.nonzero(y)]-delta)/theta
    csq=-delta/theta
    
    # CDFysq=sp.stats.gamma.fit(ysq,floc=0,fscale=1)    
    
    Fysq=sp.stats.gamma.cdf(ysq,k,scale=1)
    Fcsq=sp.stats.gamma.cdf(csq,k,scale=1)
    
    FysqkP1=sp.stats.gamma.cdf(ysq,k+1,scale=1)
    FcsqkP1=sp.stats.gamma.cdf(csq,k+1,scale=1)
    
    Fcsq2k=sp.stats.gamma.cdf(2*csq,2*k,scale=1)
    
    # CRPS calculation into two parts, the nonzeros...
    crps[np.nonzero(y)] = ysq*(2*Fysq-1) - csq*np.power(Fcsq,2) + k*( 1 + 2*Fcsq*FcsqkP1-np.power(Fcsq,2)-2*FysqkP1) - k/math.pi*betaf*(1-Fcsq2k)
    
    # and the zeros...
    crps[np.isclose(y,0.)] = csq*(2*Fcsq-1) - csq*np.power(Fcsq,2) + k*( 1 + 2*Fcsq*FcsqkP1-np.power(Fcsq,2)-2*FcsqkP1) - k/math.pi*betaf*(1-Fcsq2k)
    
    # and then recombine them...
    crps=theta*np.nanmean(crps)
    
    return 1e4*crps


#%% (4) Optimization Functions of Conditional CSGD Error Model 

# 4.1 Optimization for Regression Parameters of Conditional CSGD Model
#     (with 'covariate','nonlinear' options)

def fit_regression(outarr, climarr, p_clim, linear=True, initguess=False, constrained=False,
                   include_covariates=False, covars=False,whichcovars=[1,0,0]):
    
    # outarr is imerg; climarr is ground truth; p_clim is Climatological params
    
    if len(p_clim)!=3:
        sys.exit("something wrong in climatological CSGD parameters!")
 
      
    # Initial guess of regression parameters: when satellite obs. equals zero;
    zeroind=np.equal(outarr,0.)
    zeromean=np.nanmean(climarr[zeroind])  
    zerosig=np.nanstd(climarr[zeroind])


    # (1) for Linear Regression: 
    if linear==True:
        
        # Eqs.(3-5); Reference, p2805
        alpha2=zeromean/p_clim[0]
        alpha4=zerosig/p_clim[1]/np.sqrt(alpha2)
        
        # Initial Guesses 
        if np.all(initguess==False):
            pst=[1.0, alpha2, 0.1, alpha4, 0.0, 0.0, 0.0]
            # pst=[0.001, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
        else:
            pst=initguess
        
        # Bounds Constraints (alpha1 is not actually used in linear model)
        if constrained: 
            bnds = ( (1.0, 1.0), (0.1*alpha2, 10.*alpha2), (0.0001, 5.0), (0.1*alpha4, 10.*alpha4), 
                    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        
        else:
            bnds = ( (1.0, 1.0), (0.0001, 10.0), (0.0001, 5.0), (0.0001, 10.0), 
                    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) )
        



    # (2) for Nonlinear Regression:
    elif linear==False:
        
        al1al2bnds = ( (0.0001, 1.0), (0.0001, 10.0) )
        al1al2=[0.01, zeromean/p_clim[0]]
        
        def estimal1al2(params,zeromean,climmean):
            return np.power(zeromean-climmean/params[0]*np.log(1.+params[1]*(np.exp(params[0])-1.)),2)

        optimal1al2=sp.optimize.minimize(estimal1al2,al1al2,args=(zeromean, p_clim[0]), method='L-BFGS-B', bounds=al1al2bnds)
        alpha1=optimal1al2.x[0]
        alpha2=optimal1al2.x[1]
        alpha4=zerosig/p_clim[1]/np.sqrt(np.log(1.+alpha2*(np.exp(alpha1)-1.))/alpha1)
        
        
        # Initial Guesses 
        if np.all(initguess==False):
            pst=[alpha1, alpha2, 0.1, alpha4, 0.0, 0.0, 0.0]
            # pst=[0.001, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
        else:
            pst=initguess
        
        # Bounds Constraints
        if constrained:
            bnds = ( (0.1*alpha1, 10.*alpha1), (0.1*alpha2, 10.*alpha2), (0.0001, 5.0), (0.1*alpha4, 10.*alpha4), 
                    (0.0, 0.0), (0.0, 0.0), (0.,0.0) )  
        
        else:
            # bnds = ( (0.001, 0.1), (0.0001, 10.0), (0.0001, 5.0), (0.0001, 10.0), 
            #         (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) )
            bnds = ( (0.001, 0.5), (0.0001, 10.0), (0.0001, 5.0), (0.0001, 10.0), 
                    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) )
    
    else:
        sys.exit("specification of 'linear' is incorrect!")   
    



    # (3) for Contional CSGD Model with Covariates
    if include_covariates==True:
        
        ######
        bndstemp=list(bnds)
        
        if len(covars.shape)==1:
            covars = np.expand_dims(covars, axis=1)
        for i in range(0,covars.shape[1]):
            if whichcovars[i]==True:
                pst[4+i]=0.
                bndstemp[4+i]=(0.0001,10.0)
        bnds=tuple(bndstemp)   
        ######

       
    tempind=np.logical_and(np.isfinite(outarr),np.isfinite(climarr))
    x=outarr[tempind].astype('float64')
    climarr=climarr[tempind].astype('float64')
    xmean=np.nanmean(x)
    
    if include_covariates==True:   
        covars=covars.astype('float64')
        covars=covars[tempind,:]
        covmean=np.nanmean(covars,axis=0)
    else:
        covmean=False


    output=sp.optimize.minimize(crps_regression, pst, args=(x,xmean,climarr,p_clim,linear,include_covariates,covars,covmean), 
                                method='L-BFGS-B', options={'disp': False}, bounds=bnds)
    return output,xmean,covmean


  



# 4.2 CRPS minimization for regression parameters (with 'covariate','nonlinear' options)
def crps_regression(par, x, xmean, climx, p_clim, linear=True,
                    include_covariates=False, covars=False, covmean=False):

    # x is imerg; climx is ground truth, p_clim is Climatological params
    
    if len(x)<50:
        mu_clim=np.nan
        sigma_clim=np.nan
        delta_clim=np.nan
        sys.exit('not sure what to do!')
    
    else:
        mu_clim=p_clim[0]
        sigma_clim=p_clim[1]
        delta_clim=p_clim[2]

   
        if linear==True:
            if include_covariates:
                arg=par[1]+par[2]*x/xmean+par[4]*covars[:,0]/covmean[0]
            else:
                arg=par[1]+par[2]*x/xmean
            mu=mu_clim*arg
            
        elif linear==False:
            if include_covariates:
                logarg=par[1]+par[2]*x/xmean+par[4]*covars[:,0]/covmean[0]       
            else:   
                logarg=par[1]+par[2]*x/xmean
            mu=mu_clim/par[0]*np.log1p(np.expm1(par[0])*logarg)
        else:
            sys.exit("specification of 'linear' is incorrect!")

        sigma=par[3]*sigma_clim*np.sqrt(mu/mu_clim)



        # CRPS-based Minimization Function
        
        delta=delta_clim
        k=np.power(mu/sigma,2)        # shape 
        theta=np.power(sigma,2)/mu    # scale
        betaf=beta(0.5,0.5+k)
        ysq=(climx-delta)/theta
        csq=-delta/theta
        Fysq=sp.stats.gamma.cdf(ysq,k,scale=1)
        Fcsq=sp.stats.gamma.cdf(csq,k,scale=1)
        FysqkP1=sp.stats.gamma.cdf(ysq,k+1,scale=1)
        FcsqkP1=sp.stats.gamma.cdf(csq,k+1,scale=1)
        Fcsq2k=sp.stats.gamma.cdf(2*csq,2*k,scale=1)

        crps = ysq*(2.*Fysq-1.) - csq*np.power(Fcsq,2) + k*( 1. + 2.*Fcsq*FcsqkP1-np.power(Fcsq,2)-2.*FysqkP1) - k/math.pi*betaf*(1.-Fcsq2k)

        return 10000.*np.nanmean(theta*crps)

    
# 4.3 Generate a conditional CSGD (with 'covariate','nonlinear' options)
#     with fitted regression parameters of the conditional CSGD
    
def CondDistr(obs, xmean,p_clim, par, linear=True, include_covariates=False,
              covobs=False, covmean=False):
    
    mu_clim=p_clim[0]
    sigma_clim=p_clim[1]
    delta_clim=p_clim[2]


    if linear==True:
        if include_covariates:
            arg=par[1]+par[2]*obs/xmean+par[4]*covobs/covmean
        else:
            arg=par[1]+par[2]*obs/xmean

        mu=mu_clim*arg


    elif linear==False:
        if include_covariates:
            logarg=par[1]+par[2]*obs/xmean+par[4]*covobs/covmean
        else:
            logarg=par[1]+par[2]*obs/xmean

        mu=mu_clim/par[0]*np.log1p(np.expm1(par[0])*logarg)
        
    else:
        sys.exit("specification of 'linear' is incorrect!")      
        
    sigma=par[3]*sigma_clim*np.sqrt(mu/mu_clim)   
    delta=delta_clim

    return mu,sigma,delta


