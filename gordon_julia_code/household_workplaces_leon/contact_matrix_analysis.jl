include("./modules.jl")
include("./functions.jl")
#include("pickle.jl")
include("pickle.jl")

using DataFramesMeta
#include("./read_age_map.jl")

# Spot check some of the age distributions from Shamik's school for a few workplaces
# and schools
#
const all_df = CSV.read("all_data.csv")

schools = myunpickle("ContactMatrices/Leon/ContactMatrixSchools.pkl")
work = myunpickle("ContactMatricesA/Leon/ContactMatrixWorkplaces.pkl")

Gamma_file = "ContactMatrices/Base/Gamma_USA.csv"
gamma = Matrix(CSV.read(Gamma_file, header=false))

Ns = zeros(16)
for school_id in keys(schools)
    cm = schools[school_id]  # contact matrix
    # cm[i,j]: nb contacts between individual i and group j. What about with all groups?
    for i in 1:16
        Ns[i] = sum(cm[i,:])
    end
    if school_id == 450149119 #(Ns) > 5
        println("--------------------------")
        println("school_id: ", school_id)
        println("sum(cm[i,1:4]= ", Ns[1:4])
        avg_nb_links = cm ./ gamma
        @show cm[3:4, 3:4]
        @show gamma[3:4, 3:4]
        println("avg_nb_links = cm ./ gamma")
        @show avg_nb_links[3:4, 3:4]
    end
    #graph = createGraph(school_id, cm)
end

# Compute histogram

# Matrix reads properly. I tested by pickling using python and unpickling using Julia
# a 4x4 matrix: a(i,j) = (i-1) + 2*(j-1) in Julia
#               a(i,j) = i + 2*j in Python
a = myunpickle("a.pkl")
