# Based off the latest code written by Bryan, with a network closer to demographics
include("./modules.jl")
using Plots

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

    # Each house id can appear multiple times. I wish to change the id values to numbers between 1 and max_value
    # Create 2 databases, with missing values removed
    #   (person_id, school_id)
    #   (person_id, work_id)

    school_df = df[[:person_id, :school_id]]
    work_df = df[[:person_id, :work_id]]
    school_df = school_df[school_df.school_id .!= -1, :]
    work_df   = work_df[work_df.work_id .!= -1, :]
    #println("school_df")
    #println(first(school_df, 10))
    #return nothing, nothing, nothing

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
    #println("length(school_groups: $(length(school_groups))")
    #println("length(work_groups: $(length(work_groups))")

    println("++++++++++++++++++++++++++++++")
    # These does not appear to be -1s any longer
    println("school df")
    #println(first(school_df, 100))
    println("work df")
    #println(first(work_df, 100))

    # investigate min/max sizes in groups
    #group_sch = groups[:school_id]
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


    #println("sizes= ", sizes)
    #println(first(group_sch[1], 5))
    #return nothing, nothing, nothing


    # Create a Home Master Graph. Homes are disconnected
    # Every person is in a home

    # Create a Work Master Graph. Homes are disconnected
    # Ideally the small world parameter 0.31, should come from a distribution
    #println("starting home")
    home_graph = nothing
    work_graph = nothing
    school_graph = nothing

    # 12s, 25s, 33s
    @time work_graph = createWorkGraph(work_df, work_groups, 0.31, 0.8)
    println("finished work")

    # 24s, 1.2, 0.45, 5.9, 1.15, 1.4  (very volatile)
    @time home_graph = createHomeGraph(nrow(df), groups)
    println("finished home")
    # last argument is a weight

    @time school_graph = createWorkGraph(school_df, school_groups, 0.3, 0.7)
    println("finished school")

    # TODO: Must make all graphs weighted

    return home_graph, work_graph, school_graph
end
# 0.65, skipping over sizes > 1000
# Questions to Answer:
# 1) Why do schools have sz > 5000? In fact, why are sizes > 100,000?
#    And this is after I removed the missing data from the dataframes
@time home_graph, work_graph, school_graph = generateDemographicGraphs(p)

@show school_graph
@show work_graph
@show home_graph
graphs = [home_graph, work_graph, school_graph]
@show total_edges = sum(ne.(graphs))

# TODO:
# Check the weights on the three graph edges.
# Assign weights to graphs taken from P(w; mean, sigma)
# Run the SIR model
# Model will be α . β . γ . δ
# where α is the contact probability, β is the infectiousness,
# γ is the efficacy of masks and δ is the co-morbidity factor
# These four parameters will have their own distributions.
# Therefore, we can infer their parameters from various inference experiments.
# For that, we may have to read papers on inference. That is for later. 
