
#EXPERIMENTS WITH PARAMETRIZED Struct
mutable struct mp_Node{I,S,F}
	index::I  # :S, :I, :R
	status::S  # :S, :I, :R
	pred_inf_time::F
	rec_time::F
end
getIndex(node::mp_Node{Int,Symbol,Float64}) = node.index
getStatus(node::mp_Node{Int,Symbol,Float64}) = node.status
getPredInfTime(node::mp_Node{Int,Symbol,Float64}) = node.pred_inf_time
getRecTime(node::mp_Node{Int,Symbol,Float64}) = node.rec_time

struct nmp_Node{I,S,F}
	index::I  # :S, :I, :R
	status::S  # :S, :I, :R
	pred_inf_time::F
	rec_time::F
end
getIndex(node::nmp_Node{Int,Symbol,Float64}) = node.index
getStatus(node::nmp_Node{Int,Symbol,Float64}) = node.status
getPredInfTime(node::nmp_Node{Int,Symbol,Float64}) = node.pred_inf_time
getRecTime(node::nmp_Node{Int,Symbol,Float64}) = node.rec_time

#----
function parametrized_experiment()
	println("-------------------------------------------")
	println("\n\nParametrized struct experiments\n\n")
	G = erdos_renyi(50000, 0.0002)

	@time for i in nv(G)
		neighbors(G, i)
	end
	println("Loop over graph vertices (no allocations)\n")

	# Immutable
	nm_nodes = Array{nmp_Node,1}(undef, nv(G))
	# Mutable
	m_nodes  = Array{mp_Node,1}(undef, nv(G))

	for u in 1:nv(G)  # extremely fast
		m_nodes[u]  = mp_Node(u, :S, 0.8, 0.9)
		nm_nodes[u] = nmp_Node(u, :S, 0.8, 0.9)
	end
	println("=> non-mutable and parametrized typeof(node):  $(typeof(nm_nodes[2]))\n")
	println("=> mutable and parametrized typeof(node): $(typeof(m_nodes[2]))\n")

	@time for i in 1:nv(G) #(i,node) in enumerate(nodes)
		neighbors(G, i) #getIndex(node))
	end
	println("loop over vertices of a graph to find the neighbors, with allocations\n")
	println("Should not happen, since I am not accessing any structs\n")

	# each time node is accessed, there is a copy into node. Not by pointer
	# 50k allocs, 770 kbytes
	@time for node in m_nodes
		neighbors(G, node.index) #getIndex(node))
	end
	println("loop over m_nodes to neighbors using node.index, mutable (100k allocations)\n")

	# 50k allocs, 770 kbytes
	@time for node in nm_nodes
		neighbors(G, node.index) #getIndex(node))
	end
	println("loop over mm_nodes to neighbors node.index, immmutable (zero allocations)\n")

	function test_node(n1)
		n1.index += 17
	end

	# No allocations
	println("bef m_nodes[10].index= ", m_nodes[10].index)
	# No allocations
	@time for i in 1:nv(G)
		test_node(m_nodes[i])
	end
	println("test_node mutable, no allocations if test_node accesses node.index\n")
	println("aft m_nodes[10].index= ", m_nodes[10].index)

	try
		@time for i in 1:nv(G)
			test_node(nm_nodes[i])
		end
    catch
		println("Catch: cannot update an immutable structure")
	end
	println("test_node immmutable if test_node accesses node.index\n")
end

#parametrized_experiment()
