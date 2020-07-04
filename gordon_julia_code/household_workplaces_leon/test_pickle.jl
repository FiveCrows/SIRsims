include("pickle.jl")

#a = Pickle.myunpickle("county_data_file_combined.pkl")
a = Pickle.myunpickle("people_list_serialized.pkl")

using PyCall

def loadPickledPop(filename):
    with open(filename,'rb') as file:
        x = pickle.load(file)
    #return represented by dict of dicts
    return ({key: (vars(x[key])) for key in x})#.transpose()
# assign people to households

populace = loadPickledPop("people_list_serialized.pkl")

using DataFrames
DF = DataFrames
using CSV

filenm = "Leon/12073/people.txt"

# Prior to reading the file, I manually replaced delimeters in the header by tabs
# (ctrl-v ctrl-I) and removed all spaces from the file.
people = CSV.read(filenm, delim='\t') #, copycols=true)
