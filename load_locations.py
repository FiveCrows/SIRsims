#load location data from location files if necessary, otherwise load from csv
if False:
    locale = {}
    for type in loc_types:
        locale.update(loadPickles("{}s_list_serialized.pkl".format(type)))
        for location in locale:
            locale[location]["type"] = type# adds school, workplace, etc under 'type' key
            #locale[locale[location]['sp_id']] = locale[location] # makes the sp_id the key
            #locale.pop[location]
    df.columns = (df.loc['sp_id'][:])
    df.drop('sp_id')
    locale = df.to_dict()
    df.to_csv("./locale.csv")
