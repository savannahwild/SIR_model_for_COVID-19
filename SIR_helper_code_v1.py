from scipy.interpolate import interp1d
import numpy as np
import math

verbose = True


# KEY INFO FOR CALCULATING MORTALITY RISK


#     groups:     75+          65-74        45-64        15-44        0-14
pop_age_group = [ 0.082836082, 0.100116461, 0.258238655, 0.379493267, 0.179315536 ]
pop_mortality = [ 11.64,       3.13,        0.52,        0.03,        0           ]
N=56000000

# calculate "vaccination queue" listing population risk in order of vaccination
# we use this to calculate risk in vaccinated and unvaccinated groups given
# a certain number of vaccines carried out

n_groups = 5
vaccine_queue = []
for i in range(n_groups): 
    vaccine_queue +=  [pop_mortality[i]/100,]*int(round(pop_age_group[i]*N/1000,0))

    
# function to calculate mortality risk 
# in unvaccinated and vaccinated groups

def calc_risk_vac_unvac(n_vac, take_up, v_effD=0):

    i = int(round(n_vac/1000,0))
    frac_refuse = 1 - take_up
    frac_vac = 1 -frac_refuse
    total = len(vaccine_queue)*1000
    total_offered_vac = i*1000
    total_vac = total_offered_vac*frac_vac
    total_unvac = total - total_vac

    total_risk_vac = sum(vaccine_queue[:i])*1000*frac_vac
    total_risk_unvac = sum(vaccine_queue)*1000 - total_risk_vac

    if (total_vac!=0): av_risk_vac = total_risk_vac/total_vac
    else: av_risk_vac=0

    if (total_unvac!=0): av_risk_unvac = total_risk_unvac/total_unvac
    else: av_risk_unvac=0

    return [av_risk_vac, av_risk_unvac]


# helper function to calc risks and 
# set initial simulation conditions
# for given vaccine scenario

def vaccine_calculator(scenario=3,   # set to 0-5
                       take_up=1, # set between 0-1 
                       v_effD=0.9,   # set between 0-1
                       I0=None,
                       R0=None,
                       verbose=True,  # set to False to hide displayed info
                       split_I0=True  # if False all I0 are Is0
                      ):

    N = 56000000

    stage_info = [{"description": "stage 0: no-one vaccinated", "n_vac_offered": 0},
              {"description": "stage 1: 75+ vaccinated", "n_vac_offered": 4639000},
              {"description": "stage 2: 65+ vaccinated", "n_vac_offered": 10245000},
              {"description": "stage 3: 45+ vaccinated", "n_vac_offered": 24707000},
              {"description": "stage 4: 15+ vaccinated", "n_vac_offered": 45958000},
              {"description": "stage 5: all vaccinated", "n_vac_offered": N}]
    
    stage_dict = stage_info[scenario]
    description = stage_dict["description"]
    n_vac_offered = stage_dict["n_vac_offered"]
    
    prior_risk_D = sum(vaccine_queue)*1000/N
    
    V_risk_D_prior_to_vac, S_risk_D = calc_risk_vac_unvac(n_vac_offered, take_up=take_up)
    
    V_risk_D = V_risk_D_prior_to_vac*(1-v_effD)
    
    n_vac = int(take_up*n_vac_offered)
    n_unvac = int(N - n_vac)
    
    offered = n_vac_offered/N
    frac_vac = (n_vac_offered/N)*(take_up)
    
    s0 = None
    
    if R0 is not None and I0 is not None:
        if split_I0:
            Iv0 = I0*frac_vac
            Is0 = I0*(1-frac_vac)
        else:
            Iv0 = 0
            Is0 = I0    
        S0  = (1 - Is0 - Iv0 - R0)*(1-frac_vac)
        V0  = (1 - Is0 - Iv0 - R0)*frac_vac

        s0 = [S0,V0,Is0,Iv0,R0]
        
    if verbose:
        print(f"### Vaccine scenario: ###")
        #print(f"# modelled with:")
        #print(f"#")      
        print(f"#   {description}")
        print(f"#   with take up: {take_up}, ")
        print(f"#   and vaccine eff. against death: {v_effD} ")
        print(f"#   prior risk_D was {prior_risk_D:.5f} ")
        print(f"")
        print(f"frac_vac = {frac_vac:5.4f}")
        print(f"V_risk_D_prior_to_vac = {V_risk_D_prior_to_vac:.5f}")
        print(f"V_risk_D = {V_risk_D:.5f}")
        print(f"S_risk_D = {S_risk_D:.5f}")
        
        if s0 is not None:
            print(f"\ncalculated initial conditons:\n\n"+
                  f"  s0 = [ S0  {S0:.6f},\n"+
                  f"         V0  {V0:.6f},\n"+
                  f"         Is0 {Is0:.2e},\n"+
                  f"         Iv0 {Iv0:.2e},\n"+
                  f"         R0  {R0:.6f} ]")
        
    results = {
       "frac_vac": frac_vac,
       "S_risk_D": S_risk_D,
       "V_risk_D": V_risk_D,
       "V_risk_D_prior_to_vac": V_risk_D_prior_to_vac,
       "s0": s0 
    }
    return results




if verbose:
    print("""LOADED DISTRIBUTIONS:
     death_distribution
     hosp_adm_distribution
     hosp_discharge_distribution
     icu_adm_distribution
     icu_discharge_distribution\n""")
    
    print("""LOADED FUNCTIONS:
     get_cases()
     transformCI()
     get_deaths()
     vaccine_calculator()\n
    """)





# probabilities for transition times day by day starting day 0 up to day 61
p_Sympt_to_Hosp = [ 0,0.085296804,0.092785388,0.092054795,0.087671233,0.081461187,0.074520548,0.067214612,0.059543379,0.05260274,0.046392694,0.040182648,0.035068493,0.029954338,0.025570776,0.021917808,0.018630137,0.015707763,0.013150685,0.011324201,0.00913242,0.007671233,0.006575342,0.005479452,0.004383562,0.003652968,0.003287671,0.002557078,0.002191781,0.001461187,0.001461187,0.00109589,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
p_Hosp_to_Death = [0,0.04549675,0.082822656,0.086350975,0.095821727,0.093779016,0.082265552,0.068895079,0.057381616,0.051624884,0.041225627,0.034911792,0.028783658,0.024698236,0.021912721,0.018941504,0.01634169,0.016155989,0.01281337,0.010399257,0.010584958,0.008542247,0.006685237,0.007428041,0.005942433,0.005942433,0.006128134,0.005013928,0.004828227,0.003528319,0.004271123,0.002971216,0.002042711,0.002414113,0.002042711,0.00185701,0.002414113,0.00185701,0.002785515,0.001299907,0.001114206,0.001485608,0.001671309,0.00185701,0.001671309,0.001114206,0.000742804,0.001485608,0.001114206,0.000557103,0.000742804,0.000742804,0.000371402,0.000928505,0.000557103,0.000742804,0.000742804,0.000742804,0.000742804,0.001114206,0.000557103,0]
p_Sympt_to_ICU = [0,0.033450361,0.049200312,0.059440218,0.065925492,0.069387556,0.070460308,0.069582602,0.067144529,0.063779988,0.059440218,0.054515311,0.049395358,0.044226643,0.039155452,0.034230544,0.029695728,0.02540472,0.021650088,0.018236786,0.015213575,0.012677979,0.010337429,0.008484494,0.006924127,0.005656329,0.004486054,0.003608348,0.002828165,0.00234055,0.001755413,0.001365321,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
p_Hosp_to_Discharge = [ 0,0.062385321,0.082568807,0.088073394,0.08440367,0.078899083,0.075229358,0.064220183,0.060550459,0.056880734,0.047706422,0.036697248,0.034862385,0.027522936,0.022018349,0.025688073,0.014678899,0.018348624,0.014678899,0.011009174,0.011009174,0.011009174,0.00733945,0.00733945,0.003669725,0.005504587,0.005504587,0.001834862,0.003669725,0.003669725,0.003669725,0.003669725,0.003669725,0.001834862,0.001834862,0.001834862,0,0.003669725,0,0.001834862,0.001834862,0,0.001834862,0,9.94235E-17,0.001834862,0,0,0.001834862,0,0,0,0,0.001834862,1.24367E-15,0,0.001834862,0,0,0,0,0]
p_ICU_to_Discharge =[ 0,0.025205037,0.074500904,0.063729689,0.070910499,0.062832088,0.05744648,0.044880063,0.051163272,0.039494455,0.032313645,0.03590405,0.032313645,0.026928038,0.023337633,0.026030436,0.022440031,0.015259221,0.022440031,0.017952025,0.019747228,0.012566418,0.012566418,0.013464019,0.01436162,0.011668816,0.012566418,0.006283209,0.009873614,0.008976013,0.008976013,0.008976013,0.005385608,0.008976013,0.008976013,0.00718081,0.005385608,0.008976013,0.003590405,0.005385608,0.005385608,0.003590405,0.003590405,0.006283209,0.006283209,0.005385608,0.001795203,0.004488006,0.003590405,0.000897601,0.002692804,0.004488006,0,0.003590405,0.001795203,0.000897601,0.002692804,0,0.002692804,0.000897601,0,0]

n = 62

# calculate combined transition time
# from Symptom to death / hosp discharge / icu discharge

p_Sympt_to_Death = [0,]*n*2
for i in range(n):
    for j in range(n):
        p_Sympt_to_Death[i+j] += p_Sympt_to_Hosp[i]*p_Hosp_to_Death[j]

p_Sympt_to_ICU_to_Discharge = [0,]*n*2
for i in range(n):
    for j in range(n):
        p_Sympt_to_ICU_to_Discharge[i+j] += p_Sympt_to_ICU[i]*p_ICU_to_Discharge[j]

p_Sympt_to_Hosp_to_Discharge = [0,]*n*2
for i in range(n):
    for j in range(n):
        p_Sympt_to_Hosp_to_Discharge[i+j] += p_Sympt_to_Hosp[i]*p_Hosp_to_Discharge[j]

# store results for use in transform_cI

death_distribution = {"t": range(len(p_Sympt_to_Death)), "p": p_Sympt_to_Death}
hosp_adm_distribution = {"t": range(len(p_Sympt_to_Hosp)), "p": p_Sympt_to_Hosp}
hosp_discharge_distribution = {"t": range(len(p_Sympt_to_Hosp_to_Discharge)), "p": p_Sympt_to_Hosp_to_Discharge}
icu_adm_distribution = {"t": range(len(p_Sympt_to_ICU)), "p": p_Sympt_to_ICU}
icu_discharge_distribution = {"t": range(len(p_Sympt_to_ICU_to_Discharge)), "p": p_Sympt_to_ICU_to_Discharge}


def get_cases(t_obs, 
              cI_obs, 
              period = 1 # e.g. daily cases  = 1
                         #      weekly cases = 7
             ):
    
    # calc number of observation
    n_obs = len(t_obs)
    # create array to store cases by day
    C_obs = np.zeros(n_obs)

    # use function interp1d to allow
    # us to interpolate cumulative infectious level at 
    # any timepoint using simulation results
    fcI = interp1d(t_obs, cI_obs)
    
    # loop over observations
    for i in range(len(t_obs)):
        ti = t_obs[i]
        if ti < t_obs[0] + period:
            C_obs[i] = 0 # undefined
        else:
            C_obs[i] = fcI(ti ) - fcI(ti  - period)
    
    return C_obs

def transform_cI(t_obs, 
                 cI_obs, 
                 delay_dist = 0,   # delay or delay distribution
                 frac = 1,         # scale factor
                ):
    
    # function transforms cI in two stages
    # scales cI to cJ by factor frac or frac_fn
    # shifts cJ to cK using delay or delay distribution 
    
    # set delay distribution
    # allow single delay value (defaults to delay=0)
    if isinstance(delay_dist, float) or isinstance(delay_dist, int):
        t_dist = [delay_dist,]
        p_dist = [1,]
    else:  
        t_dist = delay_dist['t']
        p_dist = delay_dist['p']
    
    # set up frac_fn scaling fraction 
    # this allows frac to be a time dependent function
    if not callable(frac):
        frac_fn = lambda t: frac
    else:
        frac_fn = frac
    
    # calc number of observation
    n_obs = len(t_obs)
    
    # stage 1 - scale cI to cJ
    
    # create array to store transformed results
    cJ_obs = np.zeros(n_obs)
    
    # transform, allowing time variable fraction function
    cJ_obs[0] = 0    
    for i in range(1,len(t_obs)):
        cJ_obs[i] = cJ_obs[i-1] + (cI_obs[i]-cI_obs[i-1])*frac_fn(t_obs[i])
    
    # use function interp1d to allow
    # us to interpolate results at 
    # any timepoint 
    fcJ = interp1d(t_obs, cJ_obs)    
    

    # stage 2 - map cJ to cK using delay/distribution
    
    # create array to store transformed results
    K_obs = np.zeros(n_obs)

    for i in range(len(t_obs)):
        ti = t_obs[i]
        for j in range(len(t_dist)):
            delay = t_dist[j]
            p = p_dist[j]
            if (ti-delay) > t_obs[0]:
                K_obs[i] += p*fcJ(ti-delay)
    return K_obs


def get_deaths(t_obs, 
               cI_obs,
               period = 1,    # defaults to deaths/day
               delay_dist=0,  # delay distribution
               frac=1,        # mortality fraction
              ):
    # calculate cumulative deaths
    cD_obs = transform_cI(t_obs, cI_obs, delay_dist, frac)
    # calculate death counts (e.g. daily/weekly) over period
    D_obs = get_cases(t_obs, cD_obs, period=period)
    return D_obs
