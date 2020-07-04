using DataFrames, DataFramesMeta

function join_dataframes()
    df = readData()

    filename = "Leon_Formatted/households_formatted.csv"
    dfh = CSV.read(filename, delim=',')
    rename!(dfh, :longitude => :hlong)
    rename!(dfh, :latitude => :hlat)
    rename!(dfh, :sp_id => :house_id)
    joined_df = outerjoin(dfh[[:house_id,:hlat,:hlong,:hh_race,:hh_income]], df, on = :house_id)

    # Merge next DataFrame
    filename = "Leon_Formatted/schools_formatted.csv";
    dfs = CSV.read(filename, delim=',');
    rename!(dfs, :latitude => :slat);
    rename!(dfs, :longitude => :slong);
    rename!(dfs, :sp_id => :school_id);
    joined_df = outerjoin(dfs[[:school_id,:stco,:slat,:slong]], joined_df, on = :school_id)

    filename = "Leon_Formatted/workplaces_formatted.csv";
    dfw = CSV.read(filename, delim=',');
    rename!(dfw, :latitude => :wlat);
    rename!(dfw, :longitude => :wlong);
    rename!(dfw, :sp_id => :work_id);
    joined_df = outerjoin(dfw[[:work_id,:wlat,:wlong]], joined_df, on=:work_id)

    # Drop columns
    select!(joined_df, Not(:Column1))
    return joined_df
end

# ----------------------------------------------------------------------
function kidsAgeInSchool(joined_df)
    grps = groupby(joined_df, [:school_id, :age])

    # Remove rows with no school_id
    df1 = @where(joined_df, :school_id .!== -1)
    grp = groupby(df1, [:school_id, :age])
    by_age = @based_on(grp, count=length(:age))
    mean_age_df = groupby(by_age, :school_id)

    # sort GroupedDataFrame elements by age
    # Do not know a single command to do this
    for (i,g) in enumerate(mean_age_df)
        sort!(DataFrame(xx[i]),(:age))
    end

    mean_age_df = DataFrame(mean_age_df)
    @assert nrow(df1) == sum(mean_age_df.count)
    CSV.write("age_in_schools.csv", mean_age_df)
end

# ----------------------------------------------------------------------
function readData()
    filename = "Leon_Formatted/people_formatted.csv"
    df = CSV.read(filename, delim=',')
    rename!(df, :sp_id => :person_id)
    rename!(df, :sp_hh_id => :house_id)
    df1 = copy(df)

    # replace 'X' by -1 so that columns are of the same type
    replace!(df1.work_id, "X" => "-1")
    replace!(df1.school_id, "X" => "-1")

    tryparsem(Int64, str) = something(tryparse(Int64,str), missing)
    df1.school_id = tryparsem.(Int64, df1.school_id)
    df1.work_id   = tryparsem.(Int64, df1.work_id)
    return df1
end
# ----------------------------------------------------------------------

function processGQ()
    filename = "Leon_Formatted/gq_formatted.csv"
    df_gq = CSV.read(filename, delim=",")
    rename!(df_gq, :sp_id => :gq_id)
    rename!(df_gq, :longitude => :gqlong)
    rename!(df_gq, :latitude => :gqlat)
    rename!(df_gq, :persons => :gq_nb_people)
    select!(df_gq, Not(:stcotrbg))

    filename = "Leon_Formatted/gq_people_formatted.csv"
    df_gqp = CSV.read(filename, delim=",")
    rename!(df_gqp, :sp_id => :person_id)
    rename!(df_gqp, :sp_gq_id => :gq_id)
    select!(df_gqp, Not(:Column1))

    joined_df = outerjoin(df_gq, df_gqp, on = :gq_id)
    CSV.write("joined_gq.csv", joined_df)
    return joined_df
end

# ----------------------------------------------------------------------

joined_df = join_dataframes()
age_in_school_df = kidsAgeInSchool(joined_df)
joined_gq = processGQ()

joined_gq |> nrow
joined_df |> nrow
