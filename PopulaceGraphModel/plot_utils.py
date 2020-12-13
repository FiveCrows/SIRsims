import matplotlib.pyplot as plt

def plotPopulaceSIR(sim_result):
    t = sim_result.t()
    S = sim_result.S()
    I = sim_result.I()
    R = sim_result.R()
    plt.plot(t, S)
    plt.plot(t, I)
    plt.plot(t, R)
    plt.grid(True)
    #print("S= ", S)
    #print("I= ", I)
    #print("R= ", R)

