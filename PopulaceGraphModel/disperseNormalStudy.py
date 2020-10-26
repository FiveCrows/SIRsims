#purpose:
#To test the sensitivity of the populaceGraph model to normal dispersal of weights and contact matrix values.
#To test the reliability of new dispersal systems
#control test parameters:
default_env_scalars = {"school": 0.3, "workplace": 0.3, "household": 1} # base case

#the prevention measures in the workplaces
workplace_preventions = {'masking': 0.0, 'distancing': 0}
#the prevention measures in the schools
school_preventions = {'masking':0.0, 'distancing': 0}
#the prevention measures in the household
household_preventions = {'masking':0.0, 'distancing':0}
#combine all preventions into one var to easily pass during reweight and build
preventions = {'workplace': workplace_preventions, 'school': school_preventions, 'household': household_preventions}
#these values specify how much of a reduction in weight occurs when people are masked, depending on what mask they use, or how they are  distancing
prevention_reductions = {'masking': [0.1, 0.2, 0.3], 'distancing': 0.2071}# dustins values
#chances for difference possible masks
mask_probs = [0.25,0.5,0.25]
# coefficient of variation for weights
weight_cv = 0
# coefficient of variation for contact matrix values
contact_cv = 0

#gamma is the recovery rate, and the inverse of expected recovery time
gamma = 0.1
#tau is the transmission rate
tau = 0.08

# test set 1: vary weight_cv from 0 to 1 in 20 intervals, reweighting and resimulating for each
# test set 2: vary contact_cv from 0 to 1 in 20 intervals, rebuilding and resimulating for each

# hypothesis: higher coefficients of variation will reduce the incidence of infection, because the differential of
# infection chance to weight value is decreasing

# sanity check: see that the total weight of the graph remains approximately the same for each test
# sanity check: see if the total number of edges remains approximately the same for each test in test 2

# plot: total recovered rate vs weight_cv and contact_cv
# total percent change in recovered rate for each weight_cv, contact_cv for each age group
# hypothesis: groups with smaller contact will be more affected by variation.

#dump tests

#test set 3 perform test set 1 and 2 100 (100 enough?) times each, and calculate the variance of curve norms.
# plot variance vs weight_cv and contact_cv








