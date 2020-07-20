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
# 10) Add different zip codes

# TODO:
#   Assign masks and other parameters using probabilities. Even the mean infectiousness could
#   be sampled, but not for now.
#

using DataFrames, DataFramesMeta
using LightGraphs, MetaGraphs
using CSV
using Dictionaries
using BSON
# in SirProject environment
import Glob
D = Dictionaries
using Plots


const all_df = CSV.read("all_data.csv")
# Replace all missing by -1
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

#function extractSIR(k, df, nodes::Vector{Int})
function extractSIR(k, index, nodes::Vector{Int})
   # k is a key?
   #if nrow(df) == 0
   if length(index) == 0
      return null_df
   end
   return intersect(nodes, index)
end

# Add to all_df, the number of people in each workplace
work_groups = groupby(all_df, "work_id")
all_df = transform(work_groups, nrow => :wk_count)

# not sure why this is not seen inside the function
const null_df = DataFrame()

function generate_dBs(all_df::DataFrame)
   #null_df = DataFrame()
   small_df = all_df[[:school_id, :work_id, :wk_count, :age, :index]]
   df_s = @where(small_df, :school_id .!= -1)
   df_w = @where(all_df, :work_id .!= -1)
   # Person goes neither to school or to work
   df_ns_nw = @where(small_df, (:work_id .== -1) .& (:school_id .== -1))

   # person goes to school and to work
   df_s_w = @where(small_df, (:school_id .!= -1) .& (:work_id .!= -1))

   # Person has a job, does not go to school
   df_w_ns = @where(small_df, (:school_id .== -1) .& (:work_id .!= -1))

   # Person only goes to school with no job
   df_s_nw = @where(small_df, (:school_id .!= -1) .& (:work_id .== -1))

   # Let us collect the user ids of subsets of the population
   # 1) Age groups
   df_0_4   = @where(small_df, 0 .<= :age .< 5)
   df_5_18  = @where(small_df, 5 .<= :age .< 19)
   df_19_45 = @where(small_df, 19 .<= :age .< 46)
   df_46_65 = @where(small_df, 46 .<= :age .< 66)
   df_66_95 = @where(small_df, 66 .<= :age .< 96)
   # 2) All seniors; 66-inf
   df_seniors = df_66_95
   # 2) All adults; 19-65
   df_adults = vcat(df_46_65, df_66_95)
   # 3) All children: 0-18
   df_children = vcat(df_0_4, df_5_18)
   # 4) All grades K-12 (ages 4 to 18)
   df_k12 = @where(small_df, 4 .<= :age .< 13)  # revisit
   # 5) Middle-school
   df_ms = @where(small_df, 13 .<= :age .< 16) # revisit
   # 5) High-School
   df_hs = @where(small_df, 16 .<= :age .< 19) # revisit
   # 6) People staying at home with no job or school
   df_ns_nw = df_ns_nw
   # 7) People with co-morbidities
   #df_co = null_df  # cannot define this yet. No data
   # 8) All the people wearing masks
   #df_mask = null_df  # no data yet
   # 9) All the people not wearing masks
   #df_nomask = null_df # no data yet
   # People employed in business with 1-5 employees
   df_w_1_5 = @where(small_df, 0 .< :wk_count .< 6)
   # People employed in business with 6-20 employees
   df_w_6_20 = @where(small_df, 6 .<= :wk_count .< 21)
   # People employed in business with 21-100 employees
   df_w_21_100 = @where(small_df, 21 .<= :wk_count .< 101)
   # People employed in business with more than 100 employees
   df_w_101_inf = @where(small_df, 101 .<= :wk_count)

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
      #:df_co => df_co,
      #:df_mask => df_mask,
      #:df_nomask => df_nomask,
      :df_w_1_5 => df_w_1_5,
      :df_w_6_20 => df_w_6_20,
      :df_w_21_100 => df_w_21_100,
      :df_w_101_inf => df_w_101_inf
   )
   return Dictionary(dbs)
end

dbss = generate_dBs(all_df)

function generate_dbs_at_time(dbss, time_index, nodes_list)
   nodes_S,nodes_I,nodes_R = SIR_from_full_db(all_df, nodes_list[time_index])

   dbs_S = Dict()
   dbs_I = Dict()
   dbs_R = Dict()

   for key in keys(dbss)
      if nrow(dbss[key]) == 0 continue end
      index = dbss[key][:index]
      dbs_S[key] = intersect(index, nodes_S)
      dbs_I[key] = intersect(index, nodes_I)
      dbs_R[key] = intersect(index, nodes_R)
   end

   return [dbs_S, dbs_I, dbs_R]
end

# SLOWEST PART of this code. MUST ACCELERATE IT!
dbs_at_time = Dict()
lg = length(files)
max_lim = lg
for time_index in 1:max_lim #lg
   println("Time index: $(time_index)/$lg")
   # dbs_at_time[3] is a triplet of (lists of dataframes)
   @time dbs_at_time[time_index] = generate_dbs_at_time(dbss, time_index, nodes_list)
   #break
end

for i in 1:max_lim #length(files)
   if i == 1 println("-------------------------------") end
   println("i= ", i)
   @show dbs_at_time[i][2][:df_0_4] |> length
end

file_idx = 3
for key in keys(dbss)
   @show dbs_at_time[file_idx][2][key] |> length
end
nothing

# For each plot, I need
# Approximately correct. I really should get the times from the file names

function setupStructures(files, dbs_at_time)
   # file name: nodes_t=0000.0.bson  (extract time from file)
   time = []
   for file in files
      # float of the form 0.3, .3, at least one digit after the decimal point
      # No exponentiation
      m = match(r"[0-9]*\.+[0-9]+", file)
      push!(time, m.match)
   end
   time = 10 .* (collect(1:length(files)) .- 1.)
   vars_S = Dict()
   vars_I = Dict()
   vars_R = Dict()

   for key in keys(dbss)
      vars_S[key] = zeros(Int64, length(files))
      vars_I[key] = zeros(Int64, length(files))
      vars_R[key] = zeros(Int64, length(files))

      for file_idx in 1:max_lim #length(files)
         vars_S[key][file_idx] = dbs_at_time[file_idx][1][key] |> length
         vars_I[key][file_idx] = dbs_at_time[file_idx][2][key] |> length
         vars_R[key][file_idx] = dbs_at_time[file_idx][3][key] |> length
      end
   end
   return [time, vars_S, vars_I, vars_R]
end
#
time, vars_S, vars_I, vars_R = setupStructures(files, dbs_at_time)

# --------------------------------------
# Plot the results
function plotData(dbss, time, vars_S, vars_I, vars_R)
   pl = Vector{Any}(undef, length(dbss))
   lg_dbss = length(dbss)
   for (i,key) in enumerate(keys(dbss))
      ssum = sum([vars_S[key][1], vars_I[key][1], vars_R[key][1]])
      ssumi = 1. / ssum
      pl[i] = plot(time, vars_S[key].*ssumi, legend=true, label="S")
      plot!(size=(1300,1300))
      plot!(time, vars_I[key].*ssumi, label="I")
      plot!(time, vars_R[key].*ssumi, label="R")
      plot!(title=key)
   end
   plot_tuple = (pl[i] for i in 1:lg_dbss)
   p = plot(plot_tuple..., layout=lg_dbss)
   savefig("sir_plots.pdf")
   return p  # this return is responsible for displaying the plot
end

function plotAllInOne(dbss, vars_S, vars_I, vars_R)
   # Place all the curves on a single plot
   p = plot(size=(1000, 1000))

   for (i,key) in enumerate(keys(dbss))
      ssum = sum([vars_S[key][1], vars_I[key][1], vars_R[key][1]])
      ssumi = 1. / ssum
      plot!(time, vars_S[key].*ssumi, legend=false, label="S")
      plot!(time, vars_I[key].*ssumi, label="I")
      plot!(time, vars_R[key].*ssumi, label="R")
      plot!(title=key)
   end
   savefig("all_sir_plots_in_one.pdf")
   return p
end

# ----------------------------------------------------------------------
vars = (time, vars_S, vars_I, vars_R)
plotData(dbss, vars...)
plotAllInOne(dbss, vars...)

# ------------------------------------------------
