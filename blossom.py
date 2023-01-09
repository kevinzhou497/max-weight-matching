import networkx as nx
# G is a non empty, undirected graph, networkX graph
def blossom_algorithm(G):
  class Blossom:
    # init

    def leaves(self):
      for t in self.children:
        if isinstance(t, Blossom):
          yield from t.leaves()
        else:
          yield t

  # get the nodes of graph G
  nodes = list(G)


  maxweight = 0
  
