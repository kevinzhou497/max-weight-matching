import networkx as nx
from itertools import combinations, repeat

# helper function to convert the final result
def matching_dict_to_set(matching):
    """Converts matching dict format to matching set format

    Converts a dictionary representing a matching (as returned by
    :func:`max_weight_matching`) to a set representing a matching (as
    returned by :func:`maximal_matching`).

    In the definition of maximal matching adopted by NetworkX,
    self-loops are not allowed, so the provided dictionary is expected
    to never have any mapping from a key to itself. However, the
    dictionary is expected to have mirrored key/value pairs, for
    example, key ``u`` with value ``v`` and key ``v`` with value ``u``.

    """
    edges = set()
    for edge in matching.items():
        u, v = edge
        if (v, u) in edges or edge in edges:
            continue
        if u == v:
            raise nx.NetworkXError(f"Selfloops cannot appear in matchings {edge}")
        edges.add(edge)
    return edges
# 1 weight matching using the primal dual method

# G is a non empty, undirected graph, networkX graph
def blossom_algorithm(G, maxcardinality=False, weight = "weight"):
  class NoNode:
    pass
  class Blossom:
    # slots, can rewrite this
    __slots__ = ["children", "edges", "mybestedges"]

    # function for the leaves
    
    
    def leaves(self):
      # for all sub blossoms 
      for t in self.children:
        if isinstance(t, Blossom):
          yield from t.leaves()
        else:
          yield t

  # get the nodes of graph G
  nodes = list(G)
  maxWeight = 0

  # Finding the maximum edge weight
  maxweight = 0
  allintegers = True
  for i, j, d in G.edges(data = True):
    wt = d.get(weight, 1)
    if i != j and wt > maxweight:
      maxweight = wt
    allintegers = allintegers and (str(type(wt)).split("'")[1] in ("int", "long"))
    
  # If v is a matched vertex, mate[v] is its partner vertex.
  # If v is a single vertex, v does not occur as a key in mate.
  # Initially all vertices are single; updated during augmentation.
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

  # inBlossom[v] is the top-level blossom that v is in
  inBlossom = dict(zip(nodes, nodes))

  # parents[b] is its parent blossom
  # if it's top level, then parents[b] is None
  parent = dict(zip(nodes, repeat(None)))

  # bases[b] is b's base vertex
  blossomBase = dict(zip(nodes, nodes))

  # least slack edge
  bestedge = {}

  # dualvar[v] = 2*u(v) where u(v) is the v's variable in the dual optimization problem
  # initially, u(v) = maxweight / 2
  dualvar = dict(zip(nodes, repeat(maxWeight)))

  # if b is a non trivial blossom, blossomdual[b] = z(b) where z(b) is b's 
  # variable in the dual optimization problem
  blossomdual = {}
  
  # slack is the pi variables
  # same as allow edge
  zeroSlack = {}

  # newly discovered S-vertices
  queue = []

  # does not work inside blossoms
  def slack(v, w):
    return dualvar[v] + dualvar[w] - 2 * G[v][w].get(weight, 1)

  # assign the t label to top-level blossom that has vertex w
  # coming through an edge from vertex v
  def assignLabel(w, t, v):
    b = inBlossom[w]
    # is this needed
    assert label.get(w) is None and label.get(b) is None

    label[w] = label[b] = t
    if v is not None:
      labeledge[w] = labeledge[b] = (v,w)
    else:
      labeledge[w] = labeledge[b] = None
    bestedge[w] = bestedge[b] = None
    
    if t == 1:
      # this means b became an S-vertex or blossom, so add it to the queue
      if isinstance(b, Blossom):
        queue.extend(b.leaves())
    elif t == 2:
      base = base[b]
      assignLabel(mate[base], 1, base)
      
  
  def scanBlossom(v, w):
    path = []
    base = NoNode
    while v is not NoNode:
      # Look for a breadcrumb in v's blossom or put new
      b = inBlossom[v]
      if label[b] & 4:
        base = blossomBase[b]
        break
      assert label[b] == 1
      path.append(b)
      label[b] = 5
      
      if labeledge[b] is None:
        assert blossomBase[b] not in mate
        v = NoNode
      else:
        assert labeledge[b][0] == mate[blossomBase[b]]
        v = labeledge[b][0]
        b = inBlossom[v]
        assert label[b] == 2
        v = labeledge[b][0]
        
      # swap v and w so we alternate both paths
      if w is not NoNode:
        v, w = w,v
    for b in path:
      label[b] = 1
    return base
    
      
    
  # making a new blossom with base = base, through S-vertices v and w
  def addBlossom(base, v, w):
    bb = inBlossom[base] 
    bv = inBlossom[v]
    bw = inBlossom[w]

    b = Blossom()
    base[b] = base
    parent[b] = None 
    parent[bb] = b 
    
    # make list of sub blossoms and their interconnecting edge endpoints
    b.children = path = []
    b.edges = edgs = [(v,w)]
    
    # Trace back from v to the base
    while bv != bb:
      parent[bv] = b
      path.append(bv)
      edgs.ppend(labeledge[bv])
      assert label[bv] == 2 or (
        label[bv] == 1 and labeledge[bv][0] == mate[blossomBase[bv]]
      )
      
      # trace one step back
      v = labeledge[bv][0]
      bv = inBlossom[v]
      
      # Add base sub-blossom, reverse lists
      path.append(bb)
      path.reverse()
      edgs.reverse()
      
      # trace back from w to base
      while bw != bb:
        # add bw to the new blossom
        parent[bw] = b
        path.append(bw)
        edgs.append((labeledge[bw][1], labeledge[bw][0]))
        assert label[bw] == 2 or (
          label[bw] == 1 and labeledge[bw][0] == mate[blossomBase[bw]]
        )
        
        w = labeledge[bw][0]
        bw = inBlossom[w]
      
      # Set label to S
      assert label[bb] == 1
      label[b] = 1
      labeledge[b] = labeledge[bb]
      
      # set dual to 0
      blossomdual[b] = 0
      
      # relabel vertices
      for v in b.leaves():
        if label[inBlossom[v]] == 2:
          queue.append(v)
        inBlossom[v] = b
      
      # now, compute the best edges
      bestedgeto = {}
      for bv in path:
        if isinstance(bv, Blossom):
          if bv.mybestedges is not None:
            # walk this subblossom's least slack edges
            nblist = bv.mybestedges
            bv.mybestedges = None
          else:
            
            nblist = [
              (v,w) for v in bv.leaves() for w in G.neighbors(v) if v!= w
            ]
        else:
          nblist = [(bv, w) for w in G.neighbors(bv) if bv != w]
        for k in nblist:
          (i, j) = k
          if inBlossom[j] == b:
            i, j = j, i 
          bj = inBlossom[j]
          if (
            bj != b
            and label.get(bj) == 1
            and ((bj not in bestedgeto) or slack (i,j) < slack(*bestedgeto[bj]))
          ):
            bestedgeto[bj] = k
        # forget about least slack edge of subblossom
        bestedge[bv] = None
      b.mybestedges = list(bestedgeto.values())
      
      # select bestedge[b]
      mybestedge = None
      bestedge[b] = None
      for k in b.mybestedges:
        kslack = slack(*k)
        if mybestedge is None or kslack < mybestslack:
          mybestedge = k
          mybestslack = kslack
  def expandBlossom(b, endstage):
    # Convert sub blossoms into top level blossoms
    for s in b.children:
      parent[s] = None
      if isinstance(s, Blossom):
        if endstage and blossomdual[s] == 0:
          # Recursively expand this sub blossom
          expandBlossom(s, endstage)
        else:
          for v in s.leaves():
            inBlossom[v] == s
      else:
        inBlossom[s] = s
    # if we expand a T blossom  during a stage, sub blossoms 
    # must be relabeled
    if (not endstage) and label.get(b) == 2:
      entrychild = inBlossom[labeledge[b][1]]
      j = b.children.index(entrychild)
      if j & 1:
        # Start index is odd; go forward and wrap
        j -= len(b.childs)
        jstep = 1
      else:
        # Start index is even; go backward
        jstep -1
      # move along the blossom until we get to the base
      v, w = labeledge[b]
      while j != 0:
        # Relabel the T sub blossom
        if jstep == 1:
          p, q = b.edges[j]
        else:
          q, p = b.edges[j-1]
        label[w] = None 
        label[q] = None 
        assignLabel(w, 2, v)
        
        # Step to the next S sub blossom and note its forward edge
        zeroSlack[(p, q)] = zeroSlack[(q, p)] = True
        j += jstep
        if jstep == 1:
          v, w = b.edges[j]
        else:
          w, v = b.edges[j-1]
        
        # Step to the next T sub blossom
        zeroSlack[(v,w)] = zeroSlack[(w,v)] = True
        j += jstep
        while b.children[j] != entrychild:
          # examine vertices of the sub blossom to see whether it is reachable from 
          # neighboring S-vertex outside the expanding blossom
          
          bv = b.children[j]
          if label.get(bv) == 1:
            # got label S through one of its neighbors
            j += jstep 
            continue 
          if isinstance(bv, Blossom):
            for v in bv.leaves():
              if label.get(v):
                break 
          else:
            v = bv  
          
          # if the sub blossom contains a reachable vertex, assign label T
          # to the sub blossom
          if label.get(v):
            assert label[v] == 2 
            assert inBlossom[v] == bv 
            label[v] = None 
            label[mate[blossomBase[bv]]] = None
            assignLabel(v, 2, labeledge[v][0])
          j+= jstep 
      # remove expanded blossom
      label.pop(b, None)
      labeledge.pop(b, None)
      bestedge.pop(b, None)
      del parent[b]
      del blossomBase[b]
      del blossomdual[b]
      
  # swapping matched / unmatched edges over an alternating path
  def augmentBlossom(b, v):
    # bubble up through the blossom tree from vertex v to an immediate
    # sub blossom of b
    t = v 
    while parent[t] != b:
      t = parent[t]
    
    # recursively deal with first sub blossom
    if isinstance(t, Blossom):
      augmentBlossom(t, v)
    
    # Decide in which direction we go round the blossom
    i = j = b.children.index(t)
    if i & 1:
      j -= len(b.children)
      jstep = -1
    else:
      # start index is even
      jstep = -1
    
    # move along the blssom until we get to the base
    while j != 0:
      # step to the next sub blossom and augment recursively
      j += jstep
      t = b.children[j]
      if jstep == 1:
        w, x = b.edges[j]
      else:
        x, w = b.edges[j-1]
      if isinstance(t, Blossom):
        augmentBlossom(t, w)
      # step to the next sub blossom and augment recursively
      j += jstep
      t = b.children[j]
      if isinstance(t, Blossom):
        augmentBlossom(t, x)
      mate[w] = x
      mate[x] = w     
    # Rotate list of sub blossoms
    
    b.children = b.children[i:] + b.children[:i]
    b.edges = b.edges[i:] + b.edges[:i]
    blossomBase[b] = blossomBase[b.children[0]]
    assert blossomBase[b] == v
  
  # Swap matched/unmatched edges over an alternating path
  # between two single vertices. Augmenting path runs thru 
  # S-vertices v and w
  def augmentMatching(v,w):
    for (s,j) in ((v,w), (w,v)):
      # match vertex s to vertex j. Then trace back from s until we find
      # a single vertex, swapping matched and unmatched edges 
      # as we go 
      while 1: 
        # the blossom s is in 
        bs = inBlossom[s]
        assert label[bs] == 1 
        assert (labeledge[bs] is None and blossomBase[bs] not in mate)or (
          labeledge[bs][0] == mate[blossomBase[bs]]
        )
        
      # augment through the S-blososm from s to base
      if isinstance(bs, Blossom):
        augmentBlossom(bs, s)
      # Update mate[s]
      mate[s] = j 
      # Trace one step back
      if labeledge[bs] is None:
        # this is a single vertex so break
        break 
      t = labeledge[bs][0]
      bt = inBlossom[t]
      assert label[bt] == 2 
      # Trace one step back
      s, j = labeledge[bt]
      
      # augment through the T-blossom from j to base 
      assert blossomBase[bt] == t 
      if isinstance(bt, Blossom):
        augmentBlossom(bt, j)
      mate[j] = s
  def verifyOptimum():
    # might not be relevant here
    if maxcardinality: 
      vdualoffset = max(0, -min(dualvar.values()))
    else:
      vdualoffset = 0
    
    # all dual variables non negative
    assert min(dualvar.values()) + vdualoffset >= 0 
    assert len(blossomdual) == 0 or min(blossomdual.values()) >= 0
    
    # all edges have non negative slack, all matched edges have zero
    for i, j, d in G.edges(data=True):
      wt = d.get(weight, 1)
      if i == j:
        continue # ignore self loops 
      s = dualvar[i] + dualvar[j] - 2 * wt
      iblossoms = [i]
      jblossoms = [j]
      
      # what is this part about? iblossoms and jblossoms?
      while parent[iblossoms[-1]] is not None:
        iblossoms.append(parent[iblossoms[-1]])
      while parent[jblossoms[-1]] is not None:
        jblossoms.append(parent[jblossoms[-1]])
      # reverse them
      iblossoms.reverse()
      jblossoms.reverse()
      
      for (bi, bj) in zip(iblossoms, jblossoms):
        if bi != bj:
          break
        s += 2 * blossomdual[bi]
      assert s >= 0 
      if mate.get(i) == j or mate.get(j) == i:
        assert mate[i] == j and mate[j] == i
        assert s == 0
        
      #2 all single vertices have zero dual value 
      for v in nodes:
        assert (v in mate) or dualvar[v] + vdualoffset == 0
      #3 all blossoms with positive dual value are full 
      for b in blossomdual:
        if blossomdual[b] > 0:
          assert len(b.edges) % 2 == 1
          for (i, j) in b.edges[1::2]:
            assert mate[i] == j and mate[j] == i
  # main loop: continue until cant make improvements anymore
  while True: 
    # each iteration is a stage
    # a stage finds an augmenting path and uses it to improve the matching
    
    
    # remove labels from top level blossoms / vertices
    label.clear()
    labeledge.clear()
    
    # forget all about least slack edges
    bestedge.clear()
    for b in blossomdual:
      # review mybestedges
      b.mybestedges = None
      
    zeroSlack.clear()
    # clear the queue
    queue[:] = []
    
    # label single blossoms/vertices with S and put them in the queue
    for v in nodes:
      if (v not in mate) and label.get(inBlossom[v]) is None:
        assignLabel(v, 1, None)
        
    # loop until augment the matching
    augmented = 0 
    while True:
      # Each iteration is a substage 
      # tries to find augmenting path 
      # improves matching 
      # primal dual method pumps some slack of the dual variables
      
      # Continue labeling until all vertices which are reachable through an alternating path have got a label
      while queue and not augmented: 
        
        # take an S vertex from the queue
        v = queue.pop()
        # this means S blossom
        assert label[inBlossom[v]] == 1
        
        # Scan neighbors
        for w in G.neighbors(v):
          if w == v:
            continue 
          bv = inBlossom[v]
          bw = inBlossom[w]
          if bv == bw:
            # this edge is internal to a blossom
            continue
          if (v, w) not in zeroSlack:
            kslack = slack(v, w)
            if kslack <= 0:
              # edge k has zero slack -> it is allowable
              zeroSlack[(v,w)] = zeroSlack[(w,v)] = True
          if (v, w) in zeroSlack:
            if label.get(bw) is None:
              # (C1) w is a free vertex
              # label w with T and label its mate with S (R12)
              
              # assigning label 2 to the blossom with w coming through an edge from 
              # vertex v, since (v, w) has zero slack
              assignLabel(w, 2, v)
            elif label.get(bw) == 1:
              base = scanBlossom(v, w)
              if base is not NoNode:
                # found a new blossom
                addBlossom(base, v, w)
              else:
                # found an augmenting path
                augmentMatching(v, w)
                augmented = 1
                break
            elif label.get(w) is None:
              assert label[bw] == 2 
              label[w] = 2
              labeledge[w] = (v,w)
          elif label.get(bw) == 1:
            if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
              bestedge[bv] = (v,w)
          elif label.get(w) is None:
            if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
              bestedge[w] = (v,w)
      if augmented:
        break
    
      # There is no augmenting path under these constraints:
      # compute delta and reduce slack in the optimization problem
      deltatype = -1
      delta = deltaedge = deltablossom = None 
      
      # delta1: the minimum value of any vertex dual 
      if not maxcardinality:
        deltatype = 1
        delta = min(dualvar.values())
        
      # delta2: the minimum slack on any edge between an S vertex 
      # and a free vertex
      for v in G.nodes():
        if label.get(inBlossom[v]) is None and bestedge.get(v) is not None:
          d = slack(*bestedge[v])
          if deltatype == -1 or d < delta:
            delta = d
            deltatype = 2 
            deltaedge = bestedge[v]
      # delta3: half the minimum slack on any edge between a pair of s-blossoms
      for b in parent:
        if ( 
            parent[b] is None 
            and label.get(b) == 1 
            and bestedge.get(b) is not None
            ):
          kslack = slack(*bestedge[b])
          if allintegers:
            assert (kslack % 2 ) == 0
            d = kslack // 2
          else:
            d = kslack / 2.0 
          if deltatype == -1 or d < delta:
            delta = d 
            deltatype = 3
            deltaedge = bestedge[b]
      # delta4: minimum z variable of any T blossom
      for b in blossomdual:
        if (
          parent[b] is None 
          and label.get(b) == 2
          and (deltatype == -1 or blossomdual[b] < delta)
        ):
          delta = blossomdual[b]
          deltatype = 4
          deltablossom = b 
      if deltatype == -1:
        # no further improvement possible 
        assert maxcardinality 
        deltatype = 1 
        delta = max(0, min(dualvar.values()))
        
      # update dual variable values
      for v in nodes:
        if label.get(inBlossom[v]) == 1:
          # S-vertex: 2*u = 2*u - 2*delta
          dualvar[v] -= delta 
        elif label.get(inBlossom[v]) == 2:
          dualvar[v] += delta 
      for b in blossomdual:
        if parent[b] is None:
          if label.get(b) == 1:
            blossomdual[b] += delta
          elif label.get(b) == 2:
            blossomdual[b] -= delta
      
      # take action at any point where minimum delta occured
      if deltatype == 1:
        break 
      elif deltatype == 2:
        (v, w) = deltaedge 
        assert label[inBlossom[v]] == 1 
        zeroSlack[(v,w)] = zeroSlack[(w,v)] = True
        queue.append(v)
      elif deltatype == 3:
        # use least slack edge to continue the search
        (v,w) = deltaedge
        zeroSlack[(v,w)] = zeroSlack[(w,v)] = True 
        assert label[inBlossom[v]] == 1 
        queue.append(v)
      elif deltatype == 4:
        # expand least z blossom
        expandBlossom(deltablossom, False)
      
      # End of this substage
    # check matching is symmetric 
    for v in mate:
      assert mate[mate[v]] == v 
      
    # stop when no more augmenting paths
    if not augmented:
      break
  
    # expand all S blossoms which have zero dual
    for b in list(blossomdual.keys()):
      if b not in blossomdual:
        continue 
      if parent[b] is None and label.get(b) == 1 and blossomdual[b] == 0:
        expandBlossom(b, True)
  if allintegers:
    verifyOptimum()
    
  return matching_dict_to_set(mate)
        
      

                
        
      
        
        
      
        
      

    




  
