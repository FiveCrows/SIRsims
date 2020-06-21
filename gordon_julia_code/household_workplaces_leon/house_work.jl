# Based off the latest code written by Bryan, with a network closer to demographics
include("./modules.jl")
using Plots


# Generate Household and Workplace graphs (one for each)
# The idea is to apply multigraphs. Both Households and Workplace graphs
# are composed of multiple disconnected graphs, one per household and one
# per workplace
function generateDemographicGraphs(params)
    p = params
    population = p.household_size * p.households

    # symbols take 8 bytes
    work_classes = [:default, :unemployed, :school,]
    duties = [:None, :school, :work,]

    members = collect(1:100)
    group_size = 23

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
    @time df_age = ageDistribution(df)

    # Categories is a list of categories
    #column1 in each group is the person_id. I should rename the colums of the DataFrame
    # column name person_id became person
    categories = [:sp_hh_id, :work_id, :school_id]
    groups = sortPopulace(df, categories)

    # Create a Home Master Graph. Homes are disconnected
    # Every person is in a home

    # Create a Work Master Graph. Homes are disconnected
    @time work_graph = createWorkGraph(df, groups[:work_id], 0.31)
    @time home_graph = createHomeGraph(df, groups)

    # TODO: Must make all graphs weighted

    return home_graph, work_graph
end

@time home_graph, work_graph = generateDemographicGraphs(p)
