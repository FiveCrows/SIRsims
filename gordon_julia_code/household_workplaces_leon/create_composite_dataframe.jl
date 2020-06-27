using DataFrames

function DfDfh()
    df = readData()
    CSV.write("df.csv", df)
    rename!(df, :sp_id => :sp_pp_id )  # persons
    filename = "Leon_Formatted/households_formatted.csv"
    dfh = CSV.read(filename, delim=',')
    rename!(dfh, :longitude => :hlong)
    rename!(dfh, :latitude => :hlat)
    rename!(dfh, :sp_id => :sp_hh_id)
    println("df: "); println(first(df,5))
    println("dfh: "); println(first(dfh,5))
    CSV.write("dfh.csv", dfh); println(first(dfh,5))
    ddd = outerjoin(dfh[[:sp_hh_id,:hlat,:hlong]], df, on = :sp_hh_id)
    println("dfd: "); println( first(ddd,5))
    CSV.write("ddd.csv", ddd)
    return ddd
end

function DddDfs(ddd)
    filename = "Leon_Formatted/schools_formatted.csv";
    dfs = CSV.read(filename, delim=',');
    rename!(dfs, :latitude => :slat);
    rename!(dfs, :longitude => :slong);
    rename!(dfs, :sp_id => :school_id);
    CSV.write("dfs.csv", dfs);
    eee = outerjoin(dfs[[:school_id,:stco,:slat,:slong]], ddd, on = :school_id)
    CSV.write("eee.csv", eee);
    return eee
end

function EeeDfw(eee)
    filename = "Leon_Formatted/workplaces_formatted.csv";
    dfw = CSV.read(filename, delim=',');
    rename!(dfw, :latitude => :wlat);
    rename!(dfw, :longitude => :wlong);
    rename!(dfw, :sp_id => :work_id);
    CSV.write("dfw.csv", dfw);
    fff = outerjoin(dfw[[:work_id,:wlat,:wlong]], eee, on=:work_id)
    CSV.write("fff.csv", fff);
    return fff
end

ddd = DfDfh();
eee = DddDfs(ddd);
fff = EeeDfw(eee);
