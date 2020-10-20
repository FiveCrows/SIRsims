import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Take vaccines into account. 
# reduction is 0 in all cases. 
# So there are two levels of variation: 
#  1. vaccination percentate of the population 
#  2. fraction of the population wearing masks and social distancing (same values)

# Return dictionary parameter/values of first row of a DataFrame
def getParams(dfs, params):
    row = dfs.head(1)[list(params)]
    dct = {}
    values = row.values[0]
    for i,p in enumerate(params):
        dct[p] = values[i]
    return dct

#-----------------------------------------------------
df = pd.read_pickle("transformed_metadata.gz")

# Choose all elements with
# DataFrame headers
# ,sm,sd,wm,wd,red_mask,red_dist,tau,gamma,SIR,fn,ages
# Choose frames with sm = sd = 1, wm = wd = 0

group_key = ['sm','sd','wm','wd','v']
dfg = df.groupby(group_key)

keys = []
dfs = {}
for key, gr in dfg:
    keys.append(key)
    dfs[key]= dfg.get_group(key)

dct = getParams(df, group_key)
#-----------------------------------------------------

#---------------------
# Choose rows according to some criteria from the pandas table
# For example, wm=1, wd=1, sm=0, sd=0
# and then all the reductions of masks and keeping social reductions constant. 
# or create a 2D plot of max infection as a function of age somehow
#---------------------

#---------------------
# Plot a matrix of the maximum infected percentage as a function of reductions
def computeMaxInfections(age_bracket, df0):
    k = age_bracket
    nx, ny = 5, 5
    mat = np.zeros([nx,ny])   # <<<<
    mat_d = {}
    for r in df0.itertuples():  # loop over rows
        curve = r.SIR_age[k]
        v = r.v
        sm = r.sm  # same as sd, wm, wd
        mx = np.maximum(curve['S'][0], 1)  # 15,000
        maxI = r.maxI_age[k]  # 15688, 2848 (WHY THE DIFFERENCE?)
        N_age = r.N_age[k]
        mat_d[(v,sm)] = maxI / N_age

    # How to convert this information into a matrix
    masks_perc   = (0.,0.25,0.5,0.75,1.0)
    vaccine_perc   = (0.,0.25,0.5,0.75,1.0)
    for i,r1 in enumerate(vaccine_perc):     #"abscissa"
        for j,r2 in enumerate(masks_perc):   #"ordinate"
            try:
                #print("i,j= ", i,j)
                #print("   r1,f2= ", r1, r2)
                #print("   keys: ", mat_d.keys())
                mat[i,j] = mat_d[r1,r2]
            except:
                print("<<<< pass >>>>")
                print("    i,r1= ", i,r1)
                print("    j,r2= ", j,r2)
                print("   mat_d= ", mat_d.keys())
                #quit()
                pass
    return mat

#------------------------
def myImgshow(ax, mat, cmap):
    nx = ny = 5
    xx = np.linspace(0, 5, nx) / nx
    yy = np.linspace(0, 5, ny) / ny
    print("xx= ", xx)
    plt.rcParams['image.cmap'] = 'hot'

    def mycolor(val):
        return plt.get_cmap(cmap)(val)

    colors = []
    rects  = []

    for x in range(nx):
        for y in range(ny):
            rect = mpatches.Rectangle((xx[x]-.125,yy[y]-.125), .25, .25, color=mycolor(mat[x,y]))
            #print("xx,yy= ", xx[x], yy[y])
            rects.append(rect)
            colors.append(mycolor(1.5*mat[x,y]))

    #print("rects length: ", len(rects))
    pc = PatchCollection(rects) #, cmap='hot', array=None)
    pc.set(array=None, facecolors=colors)
    ax.add_collection(pc)

#------------------------
def plotMaxInfections(df0, dct, title):
    filnm = "max_infectivity"
    print("dct.items= ", dct.items())

    v_perc = set()
    m_perc = set()

    # collect vaccination and percent with mask values
    for row in df0.itertuples():
        m_perc.add(row.sm)
        v_perc.add(row.v)

    v_perc = np.asarray(v_perc) # not needed
    m_perc = np.asarray(m_perc)

    for k,v in dct.items():
        filnm += "_%s=%3.2f" % (k,v)

    title += "\n"

    count = 0
    for k,v in dct.items():
        if count == 0:
            comma = ""
        else:
            comma = ","
        count += 1
        title += comma + " %s=%3.2f" % (k,v) # sm=%2.1f, sd=%2.1f, wm=%2.1f, wd=%2.1f" % (sm,sd,wm,wd))

    matplotlib.rc('font', size=5)
    matplotlib.rc('xtick', labelsize=6) # axis tick labels
    matplotlib.rc('ytick', labelsize=6) # axis tick labels
    matplotlib.rc('axes', labelsize=8)  # axis label
    matplotlib.rc('axes', titlesize=10)  # subplot title
    matplotlib.rc('figure', titlesize=8)

    # Must call after matplotlib.rc statements
    fig, axes = plt.subplots(4,5, figsize=(10,8)) 
    axes = axes.reshape(-1)

    fig.suptitle(title)

    names_x = ['0', '.25', '.50', '.75', '1.0']  # vaccination
    names_y = ['0', '.25', '.50', '.75', '1.0']  # perc pop wearing masks and social distancing
    ticks = np.linspace(0,10,5) / 10.

    for k in range(19):
        if 1:
            mat = computeMaxInfections(k, df0)
            ax = axes[k]
            ax.xaxis.set_ticks_position('bottom')
            #ax.imshow(mat, vmin=0.0, vmax=0.6, cmap='hot', extent=[-0.10,1.10,-0.10,1.10], origin='lower')
            #ax.imshow(mat, vmin=0.0, vmax=1.0, cmap='hot', extent=[-0.10,1.10,-0.10,1.10], origin='lower')
            print("\nmat= ", mat)
            myImgshow(ax, mat, cmap='hot')
            ax.set_xlim(-0.125, 1.125)
            ax.set_ylim(-0.125, 1.125)
            nx,ny = len(names_x), len(names_y)  # not general

            for i in range(nx):
              for j in range(ny):
                  txt = "%4.2f" % mat[i,j]
                  if mat[i,j] > 0.30: col = 'black'
                  else: col = 'w'
                  text = ax.text(i*0.25, j*0.25, txt, ha="center", va="center", color=col)

            ax.set_title("k=%d" % k)
            ax.set_xlabel("% masked pop.")
            ax.set_ylabel("% vaccinaed pop.")
            ax.set_xticklabels(names_x)
            ax.set_yticklabels(names_y)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()
    plt.savefig(filnm + ".pdf")
    
plotMaxInfections(df, dct, title="Maximum infectivity by age category")
quit()
#-----------------------------------------------------
