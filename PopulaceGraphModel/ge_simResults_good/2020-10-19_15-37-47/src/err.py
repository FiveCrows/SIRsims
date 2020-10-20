Age brackets:  ['0:4', '5:9', '10:14', '15:19', '20:24', '25:29', '30:34', '35:39', '40:44', '45:49', '50:54', '55:59', '60:64', '65:69', '70:74', '75-100']
WARNING! slim = True, 90% of people are filtered out
Traceback (most recent call last):
  File "ge3_populace_study.py", line 88, in <module>
    model.build(trans_weighter, preventions, env_degrees)
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 523, in build
    self.addEnvironment(env, alg)
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 697, in addEnvironment
    alg(environment, self.environment_degrees[environment.type])
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 853, in clusterPartitionedStrogatz
    self.clusterWithMatrix( environment, avg_degree, 'strogatz')
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 935, in clusterWithMatrix
    self.clusterBipartite(environment, p_sets[i], p_sets[j], num_edges,weight_scalar=1)
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 772, in clusterBipartite
    self.addEdge(A[i], B[B_side], environment, weight_scalar)
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 601, in addEdge
    weight = self.trans_weighter.getWeight(nodeA, nodeB, environment)*weight_scalar
  File "/Users/erlebach/covid_modeling_julia/julia_code/sir-julia/SIRsims/PopulaceGraphModel/ge1_modelingToolkit.py", line 260, in getWeight
    weight = weight * (1-self.prevention_reductions["distancing"] * environment.distancing_status[(personA, personB)])
AttributeError: 'PartitionedEnvironment' object has no attribute 'distancing_status'
