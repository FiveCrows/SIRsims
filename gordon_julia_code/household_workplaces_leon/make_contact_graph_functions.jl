#
# Establish the connections between two subgraphs that satisfy a given
# categorical distribution. We model the distribution with two entries such
# that the degree average is correct. The degree averages of the two subgraphs
# are nb_edges/n1 and nb_edges/n2, which are not integer values in general.
# nb_edges: desired total number of edges between two subgraphs
# n1, n2: number of nodes in each subgraph
function edgeListProb(n1, n2, nb_edges)
    #function setup(n1,deg1,n2,deg2)
        edge_dict = Dict()
        excess_list = []

    avg_deg1 = nb_edges / n1
    avg_deg2 = nb_edges / n2
    @show avg_deg1, avg_deg2
    # We want 2000 links between both groups
    seq1, avg_deg1_approx = generateSequence(n1, avg_deg1)
    seq2, avg_deg2_approx = generateSequence(n2, avg_deg2)
    # seq1 and seq2 = list of degrees
    # Repeat each node a number of times equal to its degree (There has to be a better approach)
    # duplicte nodes
    # Because the degrees are only of two values, separated by one, there is a fast algorithm.
    deg1 = Int16(floor(avg_deg1))
    nodes1 = collect(1:length(seq1))
    nodes11 = nodes1[seq1 .== deg1]
    nodes12 = nodes1[seq1 .== (deg1+1)]
    hh1 = vcat(repeat(nodes11, deg1), repeat(nodes12, deg1+1))

    deg2 = Int16(floor(avg_deg2))
    lg = length(seq1)
    nodes2 = collect(lg+1:lg+length(seq2))
    nodes21 = nodes2[seq2 .== deg2]
    nodes22 = nodes2[seq2 .== (deg2+1)]
    hh2 = vcat(repeat(nodes21, deg2), repeat(nodes22, deg2+1))

    hh1 = hh1[randperm(length(hh1))]
    hh2 = hh2[randperm(length(hh2))]

    total_deg1 = sum(seq1)
    total_deg2 = sum(seq2)

    # Given hh1 and hh2 of different lengths, establish the links between subgraphs

    lg = minimum((length(hh1), length(hh2)))
    lgmax = maximum((length(hh1), length(hh2)))
    edge_dict = Dict()
    for i in 1:lg
        e1 = hh1[i]
        e2 = hh2[i]
        mn = minimum((e1,e2))
        mx = maximum((e1,e2))

        if haskey(edge_dict,(mn,mx))
            push!(excess_list, (mn, mx))
            continue
        end

        edge_dict[(mn,mx)] = 1
    end
    #@show count, lg
    #@show length(edge_dict)
    #@show length(excess_list)
    @assert length(edge_dict) + length(excess_list) == lg
    #@show avg_deg1, avg_deg2
    #@show 100*length(excess_list) / lgmax
    return edge_dict, excess_list
end


# Create a graph for the school.
# total number of nodes
# cmm: contact matrix
function makeGraph(N, index_range::Tuple, cmm)
    Nv = sum(N)
    g = SimpleGraph(Nv)
    g = MetaGraph(g)
    if Nv < 25 return g, [0,0] end   # <<<<< All to all connection below Nv = 25. Not done yet.

    lo, hi = index_range
    # Assign age groups to the nodes. Randomness not important
    # These are also the node numbers for each category, sorted
    age_bins = [repeat([i], N[i]) for i in lo:hi]
    cum_N = append!([0], cumsum(N))
    all_bins = []
    for i in lo:hi append!(all_bins, age_bins[i]) end
    set_prop!(g, :age_bins, all_bins)
    ddict = Dict()
    total_edges = 0

    for i in lo:hi
        for j in lo:i
            ddict = Dict()
            Nij = Int64(floor(N[i] * cmm[i,j]))
            println("i,j= $i, $j,  Nij= $Nij")

            if Nij == 0 continue end

            total_edges += Nij
            Vi = cum_N[i]+1:cum_N[i+1]
            Vj = cum_N[j]+1:cum_N[j+1]

            # Treat the case when the number of edges dictated by the
            # contact matrices is greater than the number of available edges
            # The connectivity is then cmoplete
            lg = length(Vi)
            nbe = div(lg*(lg-1), 2)
            if Vi == Vj && Nij > nbe
                Nij = nbe
            end
            count = 0
            while true
                # p ~ Vi, q ~ Vj
                # no self-edges
                # only reallocate when necessary (that would provide speedup)
                # allocate 1000 at t time
                #p = getRand(Vi, 1) # I could use memoization
                #q = getRand(Vi, 1) # I could use memoization

                p = rand(Vi, 1)[]
                q = rand(Vj, 1)[]
                if p == q continue end
                # multiple edges between p,q not allowed
                if p <  q
                    ddict[(p,q)] = 1
                else
                    ddict[(q,p)] = 1
                end
                # stop when desired number of edges is reached
                lg = length(ddict)
                if length(ddict) == Nij break end
            end
            for k in keys(ddict)
                s, d = k
                add_edge!(g, s, d)
            end
        end
    end
    degrees = degree(g)
    deg = zeros(Int64, Δ(g)+1)
    for i in 1:nv(g)
        deg[1+degrees[i]] += 1
    end
    println("Degree distribution")
    println("(N=nv(g)), deg: ")
    for i in 1:length(deg)
        println("$(deg[i]), ")
    end
    #println("")
    # N is the age distribution in the schools
    println("total edges: ", total_edges)
    return g, deg
end
nothing

# Just for testing
function makeBipartite(N1, N2, C12, C21)
    Nv = N1 + N2
    g = SimpleGraph(Nv)
    g = MetaGraph(g)
    if Nv < 25 return g, [0,0] end   # <<<<< All to all connection below Nv = 25. Not done yet.

    # Assign age groups to the nodes. Randomness not important
    # These are also the node numbers for each category, sorted
    all_bins = vcat(repeat([1], N1), repeat([2], N2))'   # row vector
    cum_N = append!([0], cumsum([N1,N2]))
    all_bins = []
    #set_prop!(g, :age_bins, all_bins)   # not clear where used
    ddict = Dict()
    total_edges = 0

    lo, hi = 1, 2
    N = [N1, N2]
    cmm = Dict()
    cmm[1,2] = C12
    cmm[2,1] = c21

    for i in lo:hi
        for j in lo:i
            ddict = Dict()
            Nij = Int64(floor(N[i] * cmm[i,j]))

            if Nij == 0 continue end

            total_edges += Nij
            Vi = cum_N[i]+1:cum_N[i+1]
            Vj = cum_N[j]+1:cum_N[j+1]

            # Treat the case when the number of edges dictated by the
            # contact matrices is greater than the number of available edges
            # The connectivity is then cmoplete
            lg = length(Vi)
            nbe = div(lg*(lg-1), 2)
            if Vi == Vj && Nij > nbe
                Nij = nbe
            end
            count = 0
            while true
                # p ~ Vi, q ~ Vj
                # no self-edges
                # only reallocate when necessary (that would provide speedup)
                # allocate 1000 at t time
                #p = getRand(Vi, 1) # I could use memoization
                #q = getRand(Vi, 1) # I could use memoization

                p = rand(Vi, 1)[]
                q = rand(Vj, 1)[]
                if p == q continue end
                # multiple edges between p,q not allowed
                if p <  q
                    ddict[(p,q)] = 1
                else
                    ddict[(q,p)] = 1
                end
                # stop when desired number of edges is reached
                lg = length(ddict)
                if length(ddict) == Nij break end
            end
            for k in keys(ddict)
                s, d = k
                add_edge!(g, s, d)
            end
        end
    end
    degrees = degree(g)
    deg = zeros(Int64, Δ(g)+1)
    for i in 1:nv(g)
        deg[1+degrees[i]] += 1
    end
    #println("(N=nv(g)), deg: ")
    #for i in 1:length(deg)
        #println("$(deg[i]), ")
    #end
    #println("")
    # N is the age distribution in the schools
    return g, deg
end


function checkContactMatrices(g, cm, index_range::Tuple)
    lo, hi = index_range
    age_bins = get_prop(g, :age_bins)
    counts = zeros(Int64, size(cm))
    for e in edges(g)
        i = src(e)
        j = dst(e)
        age_i = age_bins[i]
        age_j = age_bins[j]
        # number of edges between classes age_i and age_j
        counts[age_i, age_j] += 1
    end

    # What is the contact matrix new_cm?
    new_cm = zeros(size(cm))
    for i=lo:hi
        for j=lo:i
            new_cm[i,j] = counts[j, i] / N[i]  # i <= j
            new_cm[j,i] = counts[j, i] / N[j]  # i <= j
        end
    end

    @show "error on cmm: ", (new_cm .- cmm)
    @show "original cm: ", cmm


    # number of expected edges
    for i in lo:hi
        for j in lo:i
            c = Int64(floor(cmm[i,j]*N[i]))
            #println("i,j,c= ", c)
        end
    end

    # catalog all the edges
    # Average Degree of g
    @show Δ(g)
    deg = degree(g)
end

# The simplest approach to symmetrization is Method 1 (M1) in paper by Arregui
function reciprocity(cm, N)
    println("reciprocity, cm= ", cm)
    cmm = zeros(4,4)
    for i in 1:4
        for j in 1:4
            if N[i] == 0
                cmm[i,j] = 0
            else
                cmm[i,j] = 1/(2 * N[i])  * (cm[i,j]*N[i] + cm[j,i]*N[j])
            end
            if N[j] == 0
                cmm[j,i] = 0
            else
                cmm[j,i] = 1/(2 * N[j])  * (cm[j,i]*N[j] + cm[i,j]*N[i])
            end
        end
    end
    return cmm
end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end


function getContactMatrix(filenm, N, school_id, index_range)
	# index_range: age range of interest in the contact matrix
    lo, hi = index_range
    dict = myunpickle(filenm)
    cm = dict[school_id]
    cm = cm[lo:hi,lo:hi]   # CM for the school. (I really need the school numbers)
    cmm = reciprocity(cm, N)
    return cmm
end
