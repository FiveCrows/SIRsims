
#EXPERIMENTS WITH NON-PARAMETRIZED Struct
mutable struct mnp_Node
	index::Int  # :S, :I, :R
	status::Symbol  # :S, :I, :R
	pred_inf_time::Float64
	rec_time::Float64
end
getIndex(node::mnp_Node) = node.index
getStatus(node::mnp_Node) = node.status
getPredInfTime(node::mnp_Node) = node.pred_inf_time
getRecTime(node::mnp_Node) = node.rec_time

struct nmnp_Node
	index::Int  # :S, :I, :R
	status::Symbol  # :S, :I, :R
	pred_inf_time::Float64
	rec_time::Float64
end
getIndex(node::nmnp_Node) = node.index
getStatus(node::nmnp_Node) = node.status
getPredInfTime(node::nmnp_Node) = node.pred_inf_time
getRecTime(node::nmnp_Node) = node.rec_time

#----
function non_parametrized_experiment()
	println("\n\nNon-Parametrized struct experiments\n\n")
	G = erdos_renyi(50000, 0.0002)

	# Immutable
	m_nodes = Array{mnp_Node,1}(undef, nv(G))
	# Mutable
	nm_nodes  = Array{nmnp_Node,1}(undef, nv(G))

	@time for u in 1:nv(G)  # extremely fast
		m_nodes[u]  = mnp_Node(u, :S, 0.8, 0.9)
		nm_nodes[u] = nmnp_Node(u, :S, 0.8, 0.9)
	end
	println("Allocate mutable and immutable arrays, one per Graph vertex\n")

	@time for i in 1:nv(G) #(i,node) in enumerate(nodes)
		neighbors(G, i) #getIndex(node))
	end
	println("loop over vertices of a graph to find the neighbors, no allocations\n")

	# each time node is accessed, there is a copy into node. Not by pointer
	# 50k allocs, 770 kbytes
	@time for node in m_nodes
		neighbors(G, node.index) #getIndex(node))
	end
	println("loop over neighbors using node.index, mutable, no allocations\n")

	# 50k allocs, 770 kbytes
	@time for node in nm_nodes
		neighbors(G, node.index) #getIndex(node))
	end
	println("loop over neighbors using node.index, immutable, no allocations\n")

	function test_node(n1)
		n1.index += 17
	end

	# No allocations
	println("before m_nodes[10].index= ", m_nodes[10].index)
	# No allocations
	@time for i in 1:nv(G)
		test_node(m_nodes[i])
	end
	println("test_node mutable, no allocations if test_node accesses node.index\n")
	println("after m_nodes[10].index= ", m_nodes[10].index, ",  index changed\n")

	try
		@time for i in 1:nv(G)
			test_node(nm_nodes[i])
		end
	catch
		println("Cannot change parameter of a immutable structure")
	end
	println("test_node immmutable if test_node accesses node.index\n")
end

#non_parametrized_experiment()
