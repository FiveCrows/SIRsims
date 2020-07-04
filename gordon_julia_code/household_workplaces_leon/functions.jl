    # Generate Household and Workplace graphs (one for each)
# The idea is to apply multigraphs. Both Households and Workplace graphs
# are composed of multiple disconnected graphs, one per household and one
# per workplace
function generateDemographicGraphs(params)
    p = params
    #population = p.household_size * p.households

    # symbols take 8 bytes
    #work_classes = [:default, :unemployed, :school,]
    #duties = [:None, :school, :work,]

    # Id's are converted to integers. Missing data is -1 index
    df = readData()

    # person_id start at zero. Since I am programming in Julia, to avoid potential
    # errors, the numbering should start from 1
    df.person_id .+= 1

    # Each house id can appear multiple times. I wish to change the id values to numbers between 1 and max_value
    # Create 2 databases, with missing values removed
    #   (person_id, school_id)
    #   (person_id, work_id)

    school_df = df[[:person_id, :school_id]]
    work_df = df[[:person_id, :work_id]]
    school_df = school_df[school_df.school_id .!= -1, :]
    work_df   = work_df[work_df.work_id .!= -1, :]

    school_ids_dict = idToIndex(school_df.school_id)
    work_ids_dict   = idToIndex(work_df.work_id)

    # Replace the columns in three three databases with the respective
    # indexes
    school_df.school_id = [school_ids_dict[is] for is in school_df.school_id]
    work_df.work_id = [work_ids_dict[is] for is in work_df.work_id]

    #println("school_ids_dict")
    #for k in keys(school_ids_dict)
        #println("$k, $(school_ids_dict[k])")
    #end

    #println("school_df")
    #println(first(school_df, 10))

    #return nothing, nothing, nothing

    printSchoolDF() = for r in 1:nrow(school_df) println(school_df[r,:]) end

    # The databases now have renumbered Ids
    df_age = ageDistribution(df)

    # Categories is a list of categories
    #column1 in each group is the person_id. I should rename the colums of the DataFrame
    # column name person_id became person
    categories = [:sp_hh_id, :work_id, :school_id]
    groups = sortPopulace(df, categories)

    # Regenerate the groups
    school_groups = groupby(school_df, :school_id)
    work_groups   = groupby(work_df,  :work_id)

    # These does not appear to be -1s any longer
    println("school df")
    println("work df")

    # investigate min/max sizes in groups
    sizes = []
    for grp in school_groups
        push!(sizes, nrow(grp))
    end
    println("min/max sizes in schools: $(minimum(sizes)), $(maximum(sizes)))")

    sizes = []
    for grp in work_groups
        push!(sizes, nrow(grp))
    end
    println("min/max sizes in workplace: $(minimum(sizes)), $(maximum(sizes)))")


    # Create a Home Master Graph. Homes are disconnected
    # Every person is in a home

    # Create a Work Master Graph. Homes are disconnected
    # Ideally the small world parameter 0.31, should come from a distribution
    home_graph = nothing
    work_graph = nothing
    school_graph = nothing

    tot_nb_people = nrow(df)

    #println("========================================")
    weight = 0.8
    @time work_graph = createWorkGraph(tot_nb_people, work_df, work_groups, 0.1, weight)
    set_prop!(work_graph, :node_ids, work_df[:person_id])

    #@show work_graph
    #println("========================================")

    #@time old_work_graph = createOldWorkGraph(work_df, work_groups, 0.21, 0.8)
    #set_prop!(old_work_graph, :node_ids, work_df[:person_id])
    #@show old_work_graph
    #println("========================================")
    #println("finished work")
    #return nothing, nothing, nothing

    #println("========================================")
    weight = 0.4
    @time school_graph = createWorkGraph(tot_nb_people, school_df, school_groups, 0.1, weight)
    set_prop!(school_graph, :node_ids, school_df[:person_id])
    #println("finished school")
    #println("========================================")
    #@time school_graph = createOldWorkGraph(tot_nb_people, school_df, school_groups, 0.3, weight)
    #println("finished school")
    #println("========================================")
    #return nothing, nothing, nothing

    weight = 0.6
    @time home_graph = createHomeGraph(tot_nb_people, df, groups, weight)
    set_prop!(home_graph, :node_ids, df[:person_id])
    println("finished home")

    # TODO: Must make all graphs weighted

    return home_graph, work_graph, school_graph
end

# ---------------------------------------------------
