__author__ = 'Administrator'

def extend(d, k, v):
    n = d.copy()
    n[k] = v
    return n
def normalize(dist):
    if isinstance(dist, dict):
        keys = dist.keys()
        vals = [dist[k] for k in keys]
        normalize(vals)
        for k,v in zip(keys,vals):
            dist[k] = v
        return
    fdist = [float(d) for d in dist]
    s = sum(fdist)
    if s == 0:
        return
    fdist = [d/s for d in fdist]
    for i,d in enumerate(fdist):
        dist[i] = d
def cut(d, k):
	if isinstance(d, dict):
		n = d.copy()
		if k in n:
			del n[k]
		return n
	return [ v for v in d if v != k]
class DiscreteCPT(object):
    def __init__(self, vals, probTable):
        self.myVals = vals
        if isinstance(probTable, list) or isinstance(probTable, tuple):
            self.probTable = {(): probTable}
        else:
            self.probTable = probTable

    def values(self):
        return self.myVals

    def prob_dist(self,parentVals):
        if isinstance(parentVals, list):
            parentVals = tuple(parentVals)
        return dict([(self.myVals[i],p) for i,p in \
    				enumerate(self.probTable[parentVals])])
class BayesNode(object):
    def __init__(self, name, parents, cpt):
        self.parents = parents
        self.name = name
        self.cpt = cpt
class BayesNet(object):
    def __init__(self, nodes):
        self.variables = dict([(n.name, n) for n in nodes])
        self.roots = [n for n in nodes if not n.parents]
        self.nodes = nodes
    def enumerate_ask(self, var, e):
        vals = self.variables[var].cpt.values()
        dist = {}
        if var in e:
            for v in vals:
                dist[v] = 1.0 if e[var]==v else 0.0
                return dist
        for v in vals:
            dist[v] = self.enumerate_all(self.variables,extend(e, var, v))
        normalize(dist)
        return dist
    def enumerate_all(self, vars, e, v=None):
		if len(vars) == 0:
			return 1.0
		if v:
			Y = v
		else:
			Y = vars.keys()[0]
		Ynode = self.variables[Y]
		parents = Ynode.parents
		cpt = Ynode.cpt
		for p in parents:
			if p not in e:
				return self.enumerate_all(vars, e, p)

		if Y in e:
			y = e[Y]
			cp = cpt.prob_dist([e[p] for p in parents])[y]
			result = cp * self.enumerate_all(cut(vars,Y), e)
		else:
			result = 0
			for y in Ynode.cpt.values():
				cp = cpt.prob_dist([e[p] for p in parents])[y]
				result += cp * self.enumerate_all(cut(vars,Y),
													extend(e, Y, y))
		return result
burglary = BayesNode('Burglary', [],
						DiscreteCPT(['T','F'], [0.001, 0.9991]))
earthquake =BayesNode('Earthquake', [],
						DiscreteCPT(['T','F'], [0.002, 0.998]))

alarm = BayesNode('Alarm', ['Burglary', 'Earthquake'],
						DiscreteCPT(['T','F','U'],
							{('T','T'):[0.95, 0.01,0.04],
							('T','F'):[0.94, 0.05, 0.01],
							('F','T'):[0.29, 0.70,0.01],
							('F','F'):[0.001, 0.998,0.001]}))

john =BayesNode('JohnCalls', ['Alarm'],
						DiscreteCPT(['T','F'],
							{('T',):[0.9, 0.1],
							('F',):[0.05, 0.95],
                            ('U',):[0.5,0.5]}))
mary =BayesNode('MaryCalls', ['Alarm'],
						DiscreteCPT(['T','F'],
							{('T',):[0.7, 0.3],
							('F',):[0.01, 0.99],
                            ('U',):[0.5,0.5]}))

burglarynet = BayesNet([burglary, earthquake, alarm, john, mary])

evidence = {}
print "Prob. of earthquake:", \
			burglarynet.enumerate_ask('Earthquake', evidence)
evidence = {'JohnCalls':'T', 'MaryCalls':'T'}
print "Prob. of burglary, given John and Mary call:", \
		burglarynet.enumerate_ask(('Burglary','Earthquake'), evidence)
