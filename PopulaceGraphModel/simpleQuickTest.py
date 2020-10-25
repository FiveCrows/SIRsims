import pickle
import random
slim = False


with open("people_list_serialized.pkl", 'rb') as file:
    x = pickle.load(file)

    # return represented by dict of dicts
# renames = {"sp_hh_id": "household", "work_id": "work", "school_id": "school"} maybe later...

if slim == True:
    print("WARNING! slim = True, 90% of people are filtered out")
    populace = {}
    for key in x:
        if random.random() > 0.9:
            populace[key] = (vars(x[key]))
else:
    populace = {key: (vars(x[key])) for key in x}  # .transpose()

population = len(populace)
