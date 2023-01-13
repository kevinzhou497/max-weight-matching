import networkx as nx
from itertools import combinations, repeat
# G is a non empty, undirected graph, networkX graph
def blossom_algorithm(G, weight = "weight"):
  class Blossom:
    # slots, can rewrite this
    __slots__ = ["childs", "edges", "mybestedges"]

    # function for the leaves
    def leaves(self):
      for t in self.children:
        if isinstance(t, Blossom):
          yield from t.leaves()
        else:
          yield t

  # get the nodes of graph G
  nodes = list(G)
  maxWeight = 0

  # are we assuming integers or not?
  mate = {}

  # label.get(b) = 
  # None if b is unlabeled
  # 1 if b is an S-blossom
  # 2 if b is a T-blossom
  # label of a vertex is the label of its top-level containing blossom
  # if v is inside a T-blossom, label[v] is 2 iff v is reachable
  # from an S-vertex outside the blossom
  # labls assigned during a stage and reset during augmentation
  label = {}

  # labeledge[b] = (v,w) is the edge
  # through which b obtained its label (for top level blossom b)
  # labeledge[w] = (v,w) is an edge through which w is reachable
  # from outside the blossom if w is a vertex inside a T-blossom and label[w] == 2
  labeledge = {}

  # vertBlossom[v] is the top-level blossom that v is in
  vertBlossom = dict(zip(nodes, nodes))

  # parents[b] is its parent blossom
  # if it's top level, then parents[b] is None
  parents = dict(zip(nodes, repeat(None)))

  # bases[b] is b's base vertex
  bases = dict(zip(nodes, nodes))

  bestedge = {}

  # dualvar[v] = 2*u(v) where u(v) is the v's variable in the dual optimization problem
  # initially, u(v) = maxweight / 2
  dualvar = dict(zip(nodes, repeat(maxWeight)))

  blossomdual = {}
  zeroSlack = {}

  # newly discovered S-vertices
  queue = []

  def slack(v, w):
    return dualvar[v] + dualvar[w] - 2 * G[v][w].get(weight, 1)

  # assign the t label to top-level blossom that has vertex w

  def assignLabel(w, t, v):
    b = vertBlossom[w]
    # is this needed
    assert label.get(w) is None and label.get(b) is None

    label[w] = label[b] = t
    if v is not None:
      
  # making a new blossom with base = base, through S-vertices v and w
  def addBlossom(base, v, w):
    bb = vertBlossom[base]
    bv = vertBlossom[v]
    bw = vertBlossom[w]

    b = Blossom()
    bases[b] = base
    parents[b] = None 
    parents[bb] = b 

    




  
