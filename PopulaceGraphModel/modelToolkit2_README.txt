Written by Bryan Ashbill, 2020-10-26

Updates to modelToolkit2 

Functions and variables of the model toolkit two classes are now split 
that those for the intrinsic state of the system are part of the Environment classes, 
such as whose in the  environment, the edge picking and weight picking algorithms 
are now all part of Netbuilder. There are advantages to this
eters to the model, I would just give them a default value. 

That way the old scripts still worked, but after enough new features I start 
thinking the whole structure seems silly and wishing I had something more 
with more clear order that was much easier to use. Type errors are common in python 
and they give me hell because they cause runtime errors that  can be hard to identify 
and since python is dynamically typed, the compiler won't find them.  
Piling up giant lists of parameters in all the  function signatures is 
recipe for a trainwreck, so I tried to avoid headaches down the line with a 
cleaner structure that matches the parameters as object variables rather than as arguments.

----------------------------------------------------------------------
Parameters to be explained: 


prevention_prevalences:

Methods to be explained: 
model.differentiateMasks(....)

#en_type_scalars is used to scale each weight, depending on the type of environment the edge will be placed in
env_type_scalars = {"household": 1, "school": 0.3, "workplace": 0.3}

prevention_efficacies = {"masking": 0.7, "distancing": 0.7}
#this dict is used to decide who is masking, and who is distancing
