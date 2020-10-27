Written by Bryan Azbill, 2020-10-26

Updates to modelToolkit2 

Functions and variables of the model toolkit two classes are now split 
that those for the intrinsic state of the system are part of the Environment classes, 
such as whose in the  environment, the edge picking and weight picking algorithms 
are now all part of Netbuilder.

Many updates where applied to modelToolkit by including new parameters with default values.
this way, the old scripts still worked, but after enough new features
 the whole structure began to seem silly,  something
with more clear order that was much easier to understand was desirable. Type errors are common in python
and they can cause hell because they cause runtime errors that  can be hard to identify
and since python is dynamically typed, the compiler won't find them.  
Piling up giant lists of parameters in all the  function signatures is 
recipe for a trainwreck, so I tried to avoid headaches down the line with a 
cleaner structure that matches the parameters as object variables rather than as arguments.

parameters:

prevention_prevalences:
    Prevention_prevalences describes, for schools, workplaces, and homes,
    the rate at which people are practicing either distancing or masking.
    For example, prevention_prevalences["workplace"]["masking"] = 0.75 means three quarters of people will be wearing masks

differentiateMasks is a new function of the PopulaceGraph that allows a scripter to randomly differentiate between mask types among
the populace. That is, if mask_probs is a list, which sums to one, it can be passed as the sole argument to differentiateMasks,
and differentiateMasks will draw a mask type for each person, depending on the probabilities of the list

#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 0.3, "workplace": 0.3}

cv_dict is a new parameter of the NetBuilder class, which allows the user to specify noise for adding to variables.

        :param cv_dict: dict
        the cv dict allows the user to specify values for keys "weight", "contact", and "mask_eff",
        which will be used as the coefficient of variation for applying noise to these parameters, by multiplication
        of a gaussian draw centered around 1, and the variance being the cv
        noise to the weights, the number of contacts in structured environments, and the efficacy of masks
        """
