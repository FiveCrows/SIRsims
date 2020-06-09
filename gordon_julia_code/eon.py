# Graph-based python model with fast_SIR using EoN
# Objective: benchmark and test against Julia code

import EoN
import networkx as nx
import matplotlib.pyplot as plt

N = 10**5
N = 2000
rho = 0.05  # initial fraction infected
tau = 0.3   # transmission rate
gamma = 1.0 # recovery rate
#tau = 3   # transmission rate
#gamma = 1.0 # recovery rate
print("tau= %f, gamma= %f" % (tau, gamma))

kave = 25 # expected number of partners
print("generating graph G with {} nodes".format(N))
G = nx.fast_gnp_random_graph(N, kave/(N-1)) #Erdo’’s-Re’nyi graph
nb_edges = len(G.edges)
print("Initial number of infected: %f, fraction infected: %f" % (rho*N, rho))
print("graph has {} edges".format(nb_edges))

print("doing event-based simulation")
t1, S1, I1, R1 = EoN.fast_SIR(G, tau, gamma, rho=rho)

#print("doing Gillespie simulation")
#t2, S2, I2, R2 = EoN.Gillespie_SIR(G, tau, gamma, rho=rho)

print("done with simulations, now plotting")
plt.plot(t1, I1, label = "fast_SIR")
#plt.plot(t2, I2, label = "Gillespie_SIR")
plt.plot(t1, R1)
#plt.plot(t2, R2)
plt.plot(t1, S1)
#plt.plot(t2, S2)
plt.xlabel("$t$")
plt.xlabel("Number infected")
plt.legend()
plt.show()

