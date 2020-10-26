Updates to modelToolkit2:
 functions and variables of the model toolkit two classes are now split that those for the intrinsic state of the system are part of the Environment classes, such as whose in the  environment, the edge picking and weight picking algorithms are now all part of Netbuilder. There are advantages to this
10:48
for one, I'm able to use the polymorphism between Environments and StructuredEnvironments to distinguish differences between using every edge for the households and picking edges for the schools and environments
10:49
for another, later, if I want to compare the results of different net building algorithms, say, random vs strogatz, I can use polymorphism there, by exending with, say a strogatznetbuilder class, as opposed to using if statements
10:51
and hopefully, much of the code should look a little cleaner in general. Less nesting, fewer arguments that need to get passed down from function to function. Easier to add additional features.
10:51
as far as the diferentiated masks, now,
10:51
instead of taking a single int
10:51
or I mean,
10:52
a single double to represent mask effectiveness, it can take a list of doubles, to represent effictiveness for different mask types.
10:53
As was your request, the masks are drawn on the populace at whole, not just the environment, so, if a person has an school and a workplace, they will use the same mask for both
10:53
The scipt I edited with to test the new updates is ModelToolkitReworkTesting.py
10:55
before, when I needed to add parameters to the model, I would just give them a default value. That way the old scripts still worked, but after enough new features I start thinking the whole structure seems silly and wishing I had something more with more clear order that was much easier to use. Type errors are common in python and they give me hell because they cause runtime errors that  can be hard to identify and since python is dynamically typed, the compiler won't find them.  Piling up giant lists of parameters in all the  function signatures is recipe for a trainwreck, so I tried to avoid headaches down the line with a cleaner structure that matches the parameters as object variables rather than as arguments.