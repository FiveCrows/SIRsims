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

    school_ids_dict = idToIndex(school_df.school_id)
    work_ids_dict   = idToIndex(work_df.work_id)

    # Replace the columns in three three databases with the respective
    # indexes
    school_df.school_id = [school_ids_dict[is] for is in school_df.school_id]
    work_df.work_id = [work_ids_dict[is] for is in work_df.work_id]

    printSchoolDF() = for r in 1:nrow(school_df) println(school_df[r,:]) end

    # The databases now have renumbered Ids
    df_age = ageDistribution(df)

    # Categories is a list of categories
    #column1 in each group is the person_id. I should rename the colums of the DataFrame
    # column name person_id became person
    categories = [:sp_hh_id, :work_id, :school_id]
    groups = sortPopulace(df, categories)

    # Create a Home Master Graph. Homes are disconnected
    # Every person is in a home

    # Create a Work Master Graph. Homes are disconnected
    # Ideally the small world parameter 0.31, should come from a distribution
    #println("starting home")
    @time home_graph = createHomeGraph(nrow(df), groups)
    #println("finished home")
    # last argument is a weight
    #println("starting work")
    @time work_graph = createWorkGraph(df, groups[:work_id], 0.31, 0.8)
    #println("finished work")
    #println("starting school")
    @time school_graph = createWorkGraph(df, groups[:school_id], 0.3, 0.7)
    #println("finished school")

    # TODO: Must make all graphs weighted

    return home_graph, nothing, nothing
    return home_graph, work_graph, school_graph
end

# WEIGHTED GRAPHS TAKE FOREVER!!! TOTALLY INEFFICIENT. I must be using them wrong.
@time home_graph = generateDemographicGraphs(p)
@time home_graph, work_graph, school_graph = generateDemographicGraphs(p)
@show school_graph
@show work_graph
@show home_graph
graphs = [home_graph, work_graph, school_graph]
@show total_edges = sum(ne.(graphs))
