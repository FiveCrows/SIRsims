# Read dataframe with the population of Leon County.
# Make plots of SIR for different subsegments of the population:
# Consider the following subsegments:
# 1) Age groups
# 2) All adults
# 3) All children
# 4) All grades K-12
# 5) All grades 13-18
# 6) People staying at home with no job or school
# 7) People with co-morbidities
# 8) All the people wearing masks
# 9) All the people not wearing masks
#

using DataFrames, DataFramesMeta
using LightGraphs, MetaGraphs
using CSV
using Dictionaries
using BSON
import Glob
D = Dictionaries
#using Plots

const all_df = CSV.read("all_data.csv")
# Replace all missing by -1
const all_df = CSV.read("all_data.csv")
# Assume that all missings are integers (I do not know that to be true)
all_df = coalesce.(all_df, -1)
all_df.index = collect(1:nrow(all_df))
person_ids = all_df[:person_id] |> collect

# Read on all the nodes files in the current directory
# ordered alphabetically
files = Glob.glob("*bson")
nodes_list = []    # vector of Vector of
for file in files
   push!(nodes_list, BSON.load(file)[:nodes])
end
nothing

# nodes_S: list of indexes that identify individual people
# Return 3 lists: the node index for S, I, and R states
function SIR_from_full_db(all_df, nodes)
   df_index = all_df.index
   nodes_S = df_index[nodes .== 1]
   nodes_I = df_index[nodes .== 2]
   nodes_R = df_index[nodes .== 3]
   nodes_SIR = [nodes_S, nodes_I, nodes_R]
end

function extractSIR(k, df, nodes::Vector{Int})
   if nrow(df) == 0
      return null_df
   end
   println("enter, k= ", k)
   dfn = DataFrame([nodes], [:index])
   println(first(dfn,1))
   println(first(df,1))
   println("dfn = ", names(dfn))
   println("df= ", names(df))
   innerjoin(dfn, df, on=:index)
end

ss = SIR_from_full_db(all_df, nodes_list[1])

extractSIR(dbss[:df_s], nodes_I)
extractSIR.(dbss[:df_s], [nodes_S, nodes_I, nodes_R])
extractSIR(:df_s, dbss[:df_s], nodes_S)
db_S = Dictionary(Dict(k => extractSIR(k, dbss[k], nodes_S) for k in keys(dbss)))
db_I = Dictionary(Dict(k => extractSIR(k, dbss[k], nodes_I) for k in keys(dbss)))
db_R = Dictionary(Dict(k => extractSIR(k, dbss[k], nodes_R) for k in keys(dbss)))
nothing

function generate_dBs(all_df::DataFrame)
   df_s = @where(all_df, :school_id .!= -1)
   df_w = @where(all_df, :work_id .!= -1)
   # Person goes neither to school or to work
   df_ns_nw = @where(all_df, (:work_id .== -1) .& (:school_id .== -1))

   # person goes to school and to work
   df_s_w = @where(all_df, (:school_id .!= -1) .& (:work_id .!= -1))

   # Person has a job, does not go to school
   df_w_ns = @where(all_df, (:school_id .== -1) .& (:work_id .!= -1))

   # Person only goes to school with no job
   df_s_nw = @where(all_df, (:school_id .!= -1) .& (:work_id .== -1))

   null_df = DataFrame()
   # Let us collect the user ids of subsets of the population
   # 1) Age groups
   df_0_4 = null_df
   df_5_18 = null_df
   df_19_45 = null_df
   df_46_65 = null_df
   df_66_95 = null_df
   # 2) All seniors; 66-inf
   df_seniors = df_66_95
   # 2) All adults; 19-65
   df_adults = vcat(df_46_65, df_66_95)
   # 3) All children: 0-18
   df_children = vcat(df_0_4, df_5_18)
   # 4) All grades K-12 (ages 4 to 18)
   df_k12 = null_df
   # 5) Middle-school
   df_ms = null_df
   # 5) High-School
   df_hs = null_df
   # 6) People staying at home with no job or school
   df_ns_nw = df_ns_nw
   # 7) People with co-morbidities
   df_co = null_df  # cannot define this yet. No data
   # 8) All the people wearing masks
   df_mask = null_df  # no data yet
   # 9) All the people not wearing masks
   df_nomask = null_df # no data yet
   # People employed in business with 1-20 employees
   df_w_1_20 = null_df
   # People employed in business with 21-100 employees
   df_w_21_100 = null_df
   # People employed in business with more than 100 employees
   df_w_101_inf = null_df

   dbs = Dict(
      :df_ns_nw => df_ns_nw,
      :df_s_w => df_s_w,
      :df_w_ns => df_w_ns,
      :df_s_nw => df_s_nw,
      :df_s => df_s,
      :df_w => df_w,
      :df_0_4 => df_0_4,
      :df_5_18 => df_5_18,
      :df_19_45 => df_19_45,
      :df_46_65 => df_46_65,
      :df_66_95 => df_66_95,
      :df_seniors => df_seniors,
      :df_adults => df_adults,
      :df_children => df_children,
      :df_k12 => df_k12,
      :df_ms => df_ms,
      :df_hs => df_hs,
      :df_ns_nw => df_ns_nw,
      :df_co => df_co,
      :df_mask => df_mask,
      :df_nomask => df_nomask,
      :df_w_1_20 => df_w_1_20,
      :df_w_21_100 => df_w_21_100,
      :df_w_101_inf => df_w_101_inf
   )
   return Dictionary(dbs)
endnrow(df)

dbss = generate_dBs(all_df)

extractOne(nodes) = extractSIR(df, nodes)
ddd = extractOne.([nodes_S, nodes_I, nodes_R]);
ddd = extractOne(nodes_S);

function extractSIR(df_s, nodes_R)

counts = nrow.(dbss)
nothing

   #----------------------------------------------------------------------


const df_s = @where(all_df, :school_id .!= -1)
const df_w = @where(all_df, :work_id .!= -1)

# Person goes neither to school or to work
const df_ns_nw = @where(all_df, (:work_id .== -1) .& (:school_id .== -1))

# person goes to school and to work
const df_s_w = @where(all_df, (:school_id .!= -1) .& (:work_id .!= -1))

# Person has a job, does not go to school
const df_w_ns = @where(all_df, (:school_id .== -1) .& (:work_id .!= -1))

# Person only goes to school with no job
const df_s_nw = @where(all_df, (:school_id .!= -1) .& (:work_id .== -1))

db = Dictionary(Dict(:all_df => all_df,
          :df_ns_nw => df_ns_nw,
          :df_s_w => df_s_w,
          :df_w_ns => df_w_ns,
          :df_s_nw => df_s_nw,
          :df_s => df_s,
          :df_w => df_w,
      )
)


count = nrow.(db)
println(count)

null_df = DataFrame()
# Let us collect the user ids of subsets of the population
# 1) Age groups
df_0_4 = null_df
df_5_18 = null_df
df_19_45 = null_df
df_46_65 = null_df
df_66_95 = null_df
# 2) All seniors; 66-inf
df_seniors = df_66_95
# 2) All adults; 19-65
df_adults = vcat(df_46_65, df_66_95)
# 3) All children: 0-18
df_children = vcat(df_0_4, df_5_18)
# 4) All grades K-12 (ages 4 to 18)
df_k12 = null_df
# 5) Middle-school
df_ms = null_df
# 5) High-School
df_hs = null_df
# 6) People staying at home with no job or school
df_ns_nw = df_ns_nw
# 7) People with co-morbidities
df_co = null_df  # cannot define this yet. No data
# 8) All the people wearing masks
df_mask = null_df  # no data yet
# 9) All the people not wearing masks
df_nomask = null_df # no data yet
# People employed in business with 1-20 employees
df_w_1_20 = null_df
# People employed in business with 21-100 employees
df_w_21_100 = null_df
# People employed in business with more than 100 employees
df_w_101_inf = null_df

dbs = Dict(
   :df_ns_nw => df_ns_nw,
   :df_s_w => df_s_w,
   :df_w_ns => df_w_ns,
   :df_s_nw => df_s_nw,
   :df_s => df_s,
   :df_w => df_w,
   :df_0_4 => df_0_4,
   :df_5_18 => df_5_18,
   :df_19_45 => df_19_45,
   :df_46_65 => df_46_65,
   :df_66_95 => df_66_95,
   :df_seniors => df_seniors,
   :df_adults => df_adults,
   :df_children => df_children,
   :df_k12 => df_k12,
   :df_ms => df_ms,
   :df_hs => df_hs,
   :df_ns_nw => df_ns_nw,
   :df_co => df_co,
   :df_mask => df_mask,
   :df_nomask => df_nomask,
   :df_w_1_20 => df_w_1_20,
   :df_w_21_100 => df_w_21_100,
   :df_w_101_inf => df_w_101_inf
)
dbs = Dictionary(dbs)

counts = nrow.(dbs)

#----------------------------------------------------------------------

# What is age distribution in the df_at_home
ddfg = groupby(df_at_home, :age)
ddfyoung = @where(df_at_home, (:age .< 18))
ddfold = @where(df_at_home, (:age .> 65))
