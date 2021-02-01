import pandas as pd
import glob

# States
S=0
L=1
IA=2
PS=3
IS=4
HOME=5
H=6
ICU=7
R=8
# Potentially infected
PotL=10
# People Vaccinated get one of the following two states
V1=11


# Preprocess state_stats.csv to speed up future plotting

def breakupFile(folder):
    filenm = folder+"transition_stats.csv"
    df = pd.read_csv(filenm)
    df['time_interval'] = df['to_time'] - df['from_time']
    from_time = df['from_time'].values
    df['from_day'] = [int(i) for i in from_time]
    
    # I should do this only for specific transitions
    days = df.groupby(['from_day', 'from_state', 'to_state']).agg({'time_interval': ['mean','count']})
    days.columns = ['_'.join(col).strip() for col in days.columns.values]
    
    # move "from_day" from  index to column
    days1 = days.reset_index('from_day')
    from_to = days1.groupby(["from_state", "to_state"])

    #--------------------
    # calculate prevalence: number of infected each day
    # Of interest: 'time_interval', which is averaged over an entire day

    df_IS_L = df[(df['from_state'] == IS) & (df['to_state'] == L)]
    df2 = df_IS_L.groupby(['from_id']).agg({'from_day':['mean','count']}).reset_index()
    df2.columns = ['from_id', 'from_day', 'R_indiv']
    df_Ravg = df2.groupby('from_day').agg({'R_indiv':['mean','count']}).reset_index()
    # calculate averate daily reproduction number
    df_Ravg.columns = ['from_day', 'R_avg', 'R_count']
    print("folder= ", folder)
    df_Ravg.to_csv(folder+"R_avg.csv", float_format="%2.3f")

    #--------------------
    
    try:
        L_IS = from_to.get_group((L,IS)).reset_index()
        L_IS.to_csv(folder+"L_IS.csv", float_format="%2.3f")
    except:
        pass
  
    try:
        IS_R = from_to.get_group((IS,R)).reset_index()
        IS_R.to_csv(folder+"IS_R.csv", float_format="%2.3f")
    except:
        pass

    try:
        IS_L = from_to.get_group((IS,L)).reset_index()
        IS_L.to_csv(folder+"IS_L.csv", float_format="%2.3f")
    except:
        pass
    
    try:
        L_L  = from_to.get_group((L,L)).reset_index()
        L_L.to_csv(folder+"L_L.csv", float_format="%2.3f")
    except:
        pass
    
#-----------------------------------------------------------
if __name__ == "__main__":
    # Assumes this file is located under data_ge/resul*%05d/
    folders = glob.glob("result*/")
    #folders = glob.glob("data_ge/run00012/resul*/")
    for folder in folders:
        print(folder)
        breakupFile(folder)
    
    quit()
    
