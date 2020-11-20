class StrogatzNetBuilder(NetBuilder):
    # NOT USED

    def netStrogatz(self, environment,  num_edges, weight_scalar = 1, subgroup = None, rewire_p = 0.1):
        """
         netStrogatz creates a strogatz net in the given environment

         :param environment: Environment object,
        where to add edges
         :param num_edges: int
         the number of edges to add to the environment
         :param subgroup: list
         optional in case only edges for select members of the environment are wanted
         :param rewire_p:
         the portion of edges to be included into the net by random
         :return:
        """

        if subgroup == None:
            members = environment.members
        else:
            members = subgroup

        #unpack params
        # if only one person, don't bother
        member_count = len(members)
        if member_count == 1:
            return

        local_k = math.floor(num_edges/member_count)*2
        remainder = num_edges - local_k*member_count/2
        if local_k >= member_count:
            self.buildDenseNet(environment)
            return

        for i in range(member_count):
            nodeA = members[i]
            for j in range(1, local_k // 2+1):
                rewireRoll = random.uniform(0, 1)
                if rewireRoll < rewire_p:
                    nodeB = members[(i + random.choice(range(member_count - 1))) % member_count]
                else:
                    nodeB = members[(i + j) % member_count]
                weight = self.getWeight(nodeA, nodeB, environment)
                environment.addEdge(nodeA, nodeB, weight)
        edgeList = self.genRandEdgeList(members, members, remainder)

        for edge in edgeList:
            weight = self.getWeight(nodeA, nodeB, environment)
            environment.addEdge(edge[0], edge[1], weight)

    def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, environment):
        """
        not finished yet.
        #written for the clusterMatrixGuidedPreferentialAttachment function

        :param nodeA:
        :param nodeB:
        :param attachments:
        :param environment:
        :return:
        """

        self.addEdge(nodeA, nodeB, environment)
        groupA = environment.id_to_partition[nodeA]
        groupB = environment.id_to_partition[nodeB]

        #grow secondary list
        #Adding B's friends to A's secondary
        for key in attachments[nodeA]["secondary"][nodeB]:
            attachments[nodeA]["secondary"][key].extend(attachments[nodeB]["secondary"][key])
        #Adding A's friends to B's secondary
        for key in attachments[nodeB]["secondary"][nodeA]:
            attachments[nodeB]["secondary"][key].extend(attachments[nodeA]["secondary"][key])

        #Adding B as secondary to A's friends
        for key in attachments[nodeA]:
            pass
        #Adding A as secondary to B's friends

            # grow primary list,
            # adding B to A, A to B
        attachments[nodeA]["primary"][groupB].append(nodeB)
        attachments[nodeB]["primary"][groupA].append(nodeA)

        #Adding A's friends to B

        #Adding B to A's friends
        #Adding A to B's friends
        #try:
            #attachments[""]

    def clusterMatrixGuidedPreferentialAttachment(self, environment, avg_contacts, prob_rand):
        cumulative_weight = sum(sum(environment.contact_matrix))
        num_people = len(environment.members)
        total_pos_edges = num_people * (num_people - 1) / 2
        total_edges = num_people * avg_contacts
        random_edges = math.round(prob_rand * total_edges)
        remaining_edges = total_edges - random_edges
        vecM = np.matrix.flatten(environment.contact_matrix)
        num_partitions = len(vecM)
        partitionAttachments = {}


        # speed up, in case there aren't many duplicates likely anyways
        random_duplicate_rate = (random_edges - 1) / total_pos_edges
        if random_duplicate_rate > 0.01:
            rand_edges = random.choices(list(itertools.combinations(environment.members, 2)), k=random_edges)
            for edge in rand_edges:
                self.addEdge(edge[0], edge[1], environment)
        else:
            for i in range(random_edges):
                sel_A = random.choice(num_people)
                sel_B = (sel_A + random.choice(num_people - 1)) % num_people
                self.addEdge(environment.members[sel_A], environment.members[sel_B], environment)

        # now adding preferential attachment edges
        partition_dist = [sum(vecM[:i] for i in range(num_partitions))] / sum(vecM)
        # partition_dist projects the edge_partition to  [0,1], such that the space between elements is in proportion to
        # the elements contact
        for i in range(remaining_edges):
            # this selects a partition element using partition_dist
            # then, from vec back to row/col
            selector = random.random()
            raw_partition = list(filter(range(num_partitions),
                                        lambda i: partition_dist[i] < (selector) & partition_dist[i + 1] > (
                                        selector)))
            partition_A = raw_partition % environment.contact_matrix.shape[0]
            partition_B = raw_partition // environment.contact_matrix.shape[0]

            def addEdgeWithAttachmentTracking(self, nodeA, nodeB, attachments, id_to_partition, mask_p, weight):
                w = self.trans_weighter.genMaskScalar(mask_p) * weight
                self.graph.addEdge(nodeA, nodeB, transmission_weight = w)
                groupA = id_to_partition[nodeA]
                groupB = id_to_partition[nodeB]

                # grow secondary list
                # Adding B's friends to A's secondary
                for key in attachments[nodeA]["secondary"][nodeB]:
                    attachments[nodeA]["secondary"][key].extend(attachments[nodeB]["secondary"][key])
                # Adding A's friends to B's secondary
                for key in attachments[nodeB]["secondary"][nodeA]:
                    attachments[nodeB]["secondary"][key].extend(attachments[nodeA]["secondary"][key])

                # Adding B as secondary to A's friends
                for key in attachments[nodeA]:
                    pass
                # Adding A as secondary to B's friends

                # grow primary list,
                # adding B to A, A to B
                attachments[nodeA]["primary"][groupB].append(nodeB)
                attachments[nodeB]["primary"][groupA].append(nodeA)

                # Adding A's friends to B

                # Adding B to A's friends
                # Adding A to B's friends
                # try:

                # attachments[""]



class prefAttachmentNetBuilder(NetBuilder):
    pass
