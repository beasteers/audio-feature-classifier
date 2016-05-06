from pomegranate import *
import numpy as np
import json, pickle, os
import matplotlib.pyplot as plt
import scipy.cluster.vq as sp


# Get list index w/out exception
def safe_get(l, i, default=None):
    try: return l[i]
    except IndexError: return default

# Gets file in current working directory
def in_cwd(fname=''):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
cwd = in_cwd()

#Print output to one line
def same_line(out):
    sys.stdout.write( '\x1b[2K{}\r'.format(out) )
    sys.stdout.flush()

# Read file as string
def read(file):
    with open(in_cwd(file), 'rU') as in_file:
        data = in_file.read()
    return data

# Read pickle from file
def read_pickle(file):
    with open(in_cwd(file), 'rb') as in_file:
        return pickle.load(in_file)

# Save file as string
def save(content, file):
    with open(in_cwd(file), 'w') as out_file:
        out_file.write(content)


# Save pickle to file
def save_pickle(data, file):
    with open(in_cwd(file), 'wb') as out_file:
        pickle.dump(data, out_file)


# Build hmm from vector file
def vector_to_hmm(name, vectorFile, k=20):
    # Load data vectors
    data = read_pickle(vectorFile)
    print data['name']
    vectors = np.array(data['vectors'])

    # Feature vector length
    m = vectors.shape[1]

    # k is number of centroids in k means
    # Get group indexes for vectors
    centroids,_ = sp.kmeans(vectors, k)
    index,_ = sp.vq(vectors, centroids)

    # Split vectors into clusters
    clusters = [vectors[index == ki] for ki in range(k)]

    # Fill Transition Matrix
    transitions = np.ones((k,k))
    for i in range(1, len(index)):
        transitions[index[i-1], index[i]] += 1

    for i in range(len(transitions)):
        transitions[i] /= sum(transitions[i])

    # Generate Emission Distributions
    distributions = []
    for i, c in enumerate(clusters):
        dist = MultivariateGaussianDistribution(centroids[i], np.eye(m))
        dist.fit(c)
        distributions.append(dist)


    return HiddenMarkovModel.from_matrix( transitions, distributions, starts=[1./k for _ in range(k)], ends=None, name=name )
'''
Sample from pomegranate:

        matrix = [ [ 0.4, 0.5 ], [ 0.4, 0.5 ] ]
		distributions = [NormalDistribution(1, .5), NormalDistribution(5, 2)]
		starts = [ 1., 0. ]
		ends = [ .1., .1 ]
		state_names= [ "A", "B" ]

		model = Model.from_matrix( matrix, distributions, starts, ends,
			state_names, name="test_model" )

'''

