
#---------------------
def generateCurves(row):
    ages = row.ages
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
        ages_l = v
        for k_age, v_age in v.items():   # 1 .. 19
            curves[k_age]['S'].append(v_age['S'])
            curves[k_age]['I'].append(v_age['I'])
            curves[k_age]['R'].append(v_age['R'])
            curves[k_age]['t'].append(k)

    return curves
