import numpy as np

#---------------------
def generateCurves(row):
    ages = row.ages
    # ages.keys: 2, 4, 6, ... : time levels
    print("ages.keys(): ", ages.keys())
    print("ages[2].keys(): ", ages[2].keys())
    print("ages[2][4].keys(): ", ages[2][4].keys())
    curves = {}
# I want curves[0-19].S = [S[0], S[10], ...]
# I want curves[0-19].I = [I[0], I[10], ...]

    for k in range(0,20):
        curves[k] = {}
        curves[k]['S'] = []
        curves[k]['I'] = []
        curves[k]['R'] = []
        curves[k]['t'] = []

    for k, v in ages.items():
        # For each time, extract SIR curves for all ages
        # k is the time level, v is the SIR data at that time
        ages_l = v
        for k_age, v_age in v.items():   # 1 .. 19
            curves[k_age]['S'].append(v_age['S'])
            curves[k_age]['I'].append(v_age['I'])
            curves[k_age]['R'].append(v_age['R'])
            curves[k_age]['t'].append(v_age['t'])

    # Convert to numpy array
    for k_age in range(0,20):
        curves[k_age]['S'] = np.asarray(curves[k_age]['S'])
        curves[k_age]['I'] = np.asarray(curves[k_age]['I'])
        curves[k_age]['R'] = np.asarray(curves[k_age]['R'])
        curves[k_age]['t'] = np.asarray(curves[k_age]['t'])

    return curves
