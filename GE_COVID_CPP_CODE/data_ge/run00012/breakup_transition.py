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
    days1.to_csv("days1.csv") #, index_col=0)
    
    days2 = days1.groupby(["from_state", "to_state"])
    
    L_IS = days2.get_group((L,IS)).reset_index()
    IS_R = days2.get_group((IS,R)).reset_index()
    IS_L = days2.get_group((IS,L)).reset_index()
    
    L_IS.to_csv(folder+"L_IS.csv", float_format="%2.3f")
    IS_R.to_csv(folder+"IS_R.csv", float_format="%2.3f")
    IS_L.to_csv(folder+"IS_L.csv", float_format="%2.3f")
    
#-----------------------------------------------------------
if __name__ == "__main__":
    folders = glob.glob("resul*/")
    for folder in folders:
        print(folder)
        breakupFile(folder)
    
    quit()
    
