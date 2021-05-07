# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:16:35 2021

@author: savan
"""
#stages vs cumulative, take up 0.9

from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.integrate import odeint
from SIR_helper_code_v1 import *

#simulation
def sdot_SVIR(s,t,params):

    #System variables: S, V, Is, Iv, R
    S,V,Is,Iv,R = s

    #model parameters: rzero(t), Ti, v_effT, v_effI
    rzero,Ti,v_effT,v_effI=params   

    # fill in rate expressions
    rate_S_Is = (rzero/Ti)*S*Is + (1-v_effT)*(rzero/Ti)*S*Iv
    rate_V_Iv = (1-v_effI)*((rzero/Ti)*V*Is + (1-v_effT)*(rzero/Ti)*V*Iv)
    rate_Is_R = Is/Ti
    rate_Iv_R = Iv/Ti

    # fill using rates calculated above
    dS  = -rate_S_Is
    dV = -rate_V_Iv
    dIv = rate_V_Iv - rate_Iv_R
    dIs = rate_S_Is - rate_Is_R
    dR =  rate_Is_R + rate_Iv_R
    ds = [dS, dV, dIs, dIv, dR]
    return ds

# define time points with 1 observation per day
t_max = 400
t_obs = np.linspace(0,t_max, t_max+1)

# define model parameters
rzero = 1.25
Ti = 12.9
v_effI = 0.67
v_effT = 0.09
params=[rzero, Ti, v_effI, v_effT]

# define I0 and R0
I0 = 402/100000
R0 = 3864000/56000000

#fraction of people who take up vac offer
#between 0 and 1
take_up = 0.78

# vaccine eff. in reducing risk of death
# between 0 and 1
v_effD = 1

# scenario number can be: 0, 1, 2, 3, 4, 5
# from no vaccination to full vaccination
scenarios = [0,1,2,3,4,5]

#calculate frac_vac, risks, initial conditions

#create figure
fig1 = plt.figure(figsize = (8, 8))
plt.style.use('ggplot')
#add 2 subplots to figure
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)

#create empty list for cumulative deaths
all_cD_obs = []
#create empty list for cumulative cases
all_cI_obs = []
for scenario in scenarios: 
    # vaccine calculator calculates
    #S0, V0, Is0, and Iv0
    results = vaccine_calculator(scenario, take_up, v_effD, I0, R0, verbose=True )

    # load calculated results
    frac_vac = results["frac_vac"]
    S_risk_D = results["S_risk_D"]
    V_risk_D = results["V_risk_D"]
    V_risk_D_prior_to_vac = results["V_risk_D_prior_to_vac"]
    s0 = results["s0"]

    # run simulation
    s_obs = odeint(sdot_SVIR, s0, t_obs, args=(params,))

    # to unpack simulation
    S_obs  = s_obs[:,0]
    V_obs  = s_obs[:,1]
    Is_obs = s_obs[:,2]
    Iv_obs = s_obs[:,3]
    R_obs  = s_obs[:,4]
    
    #population of England
    N = 56000000
    
    #calc cumulative cases
    cIs_obs = S_obs[0] - S_obs
    cIv_obs = V_obs[0] - V_obs
    
    #total cases can be found from cumulative cases
    cI_obs = N*(cIs_obs + cIv_obs)
    all_cI_obs.append(cI_obs[-1])

    #for cumulative deaths
    #found from final culmulative death values
    cDs_obs = N*transform_cI(t_obs, cIs_obs, delay_dist=death_distribution, frac=S_risk_D)
    cDv_obs = N*transform_cI(t_obs, cIv_obs, delay_dist=death_distribution, frac=V_risk_D)
    cD_obs  = cDs_obs + cDv_obs
    all_cD_obs.append(cD_obs[-1])

    print(int(cD_obs[-1]))
    print(int(cI_obs[-1]))           
ax1.plot(scenarios, all_cD_obs)
ax2.plot(scenarios, all_cI_obs)

ax1.set_title('Total number of deaths for each vaccination scenario')
ax1.set_xlabel('Scenario')
ax1.set_ylabel('Total number of deaths')

ax2.set_title('Total number of cases for each vaccination scenario')
ax2.set_xlabel('Scenario')
ax2.set_ylabel('Total number of cases')

