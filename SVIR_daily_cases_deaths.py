# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:54:58 2021

@author: savan
"""
#basic SVIR model daily cases and deaths over time

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
    dS  = - rate_S_Is
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

# scenario number can be: 0, 1, 2, 3, 4, 5
# from no vaccination to full vaccination
scenario = 1
#fraction of people who take up vac offer
#between 0 and 1
take_up = 0.78

# vaccine eff. in reducing risk of death
# between 0 and 1
v_effD = 1

#calculate frac_vac, risks, initial conditions

#create figure
fig1 = plt.figure(figsize = (8, 8))
plt.style.use('ggplot')
#add 2 subplots to figure
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)

# vaccine calculator calculates
#S0, V0, Is0, and Iv0
results = vaccine_calculator(scenario, take_up, v_effD, I0, R0, verbose=True )

# load calculated results
frac_vac = results["frac_vac"]
S_risk_D = results["S_risk_D"]
V_risk_D = results["V_risk_D"]
V_risk_D_prior_to_vac = results["V_risk_D_prior_to_vac"]
s0 = results["s0"]
    
#run simulation
s_obs = odeint(sdot_SVIR, s0, t_obs, args=(params,))
    
#to unpack simulation
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

#calc daily cases
Cs_obs = N*get_cases(t_obs, cIs_obs, period=1)
Cv_obs = N*get_cases(t_obs, cIv_obs, period=1)
C_obs  = Cs_obs + Cv_obs
#plot daily cases
ax2.plot(t_obs,C_obs,label='all cases')
ax2.plot(t_obs,Cs_obs,label='cases in susceptible population')
ax2.plot(t_obs,Cv_obs, label='cases in vaccinated population')
#calc daily deaths
Ds_obs = N*get_deaths(t_obs, cIs_obs, period=1, delay_dist=death_distribution, frac=S_risk_D)
Dv_obs = N*get_deaths(t_obs, cIv_obs, period=1, delay_dist=death_distribution, frac=V_risk_D)
D_obs  = Ds_obs + Dv_obs
#plot daily deaths
ax1.plot(t_obs,D_obs,label='deaths from all cases')
ax1.plot(t_obs,Ds_obs,label='deaths in susceptible population')
ax1.plot(t_obs,Dv_obs, label='deaths in vaccinated population')
#format plots
ax1.set_title('Daily deaths for vaccination scenario '+str(scenario))
ax1.set_xlabel('time(days)')
ax1.set_ylabel('Number of deaths')
ax1.set_xlim(1,400)
ax1.legend()

ax2.set_title('Daily cases for vaccination scenario '+str(scenario))
ax2.set_xlabel('time(days)')
ax2.set_ylabel('Number of cases')
ax2.set_xlim(1,400)
ax2.legend()