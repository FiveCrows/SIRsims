import numpy as np

#---------------------
def generateCurves(row):
    ages = row.ages
    # ages.keys(): 2, 4, 6, ... : age time levels
    #print("ages.keys(): ", ages.keys())
    # ages[2].keys(): age brackets: 0, 1, ..., 18, 19
    #print("ages[2].keys(): ", ages[2].keys())
    # ages[2][4].keys(): 'S', 'I', 'R'
    #print("ages[2][4].keys(): ", ages[2][4].keys())
    # ages[2][4]['S']: 3818 (one value of susceptibility)
    #print("ages[2][4]['S']: ", ages[2][4]['S'])

    curves = {}
    times = np.sort(list(ages.keys()))

# I want curves[0-19].S = [S[0], S[10], ...]
# I want curves[0-19].I = [I[0], I[10], ...]

    for k_age in ages[2].keys():  # list of age brackets
        curves[k_age] = {}
        curves[k_age]['S'] = []
        curves[k_age]['I'] = []
        curves[k_age]['R'] = []
        curves[k_age]['t'] = []

    #print("ages.items()= ", ages.items())

    for age_t in ages.keys():   # age SIR time levels
        ages_sir = ages[age_t]
        #print("ages_sir: ", ages_sir)
        for k,v in ages_sir.items():
            # For each time, extract SIR curves for all ages
            # k is the time level, v is the SIR data at that time
            #for k_age, v_age in v.items():   # 1 .. 19
            #for k_age, v_age in v.items():   # 1 .. 19
            curves[k]['S'].append(v['S'])
            curves[k]['I'].append(v['I'])
            curves[k]['R'].append(v['R'])
            curves[k]['t'].append(age_t)

    # Convert to numpy array
    for k_age in ages[2].keys():
        curves[k_age]['S'] = np.asarray(curves[k_age]['S'])
        curves[k_age]['I'] = np.asarray(curves[k_age]['I'])
        curves[k_age]['R'] = np.asarray(curves[k_age]['R'])
        curves[k_age]['t'] = np.asarray(curves[k_age]['t'])

    return curves
