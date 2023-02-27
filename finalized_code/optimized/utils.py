import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


@njit
def get_all_edges(nrow, ncol):
    all_edges = []
    for i in range(0, nrow):
        for j in range(1, ncol + 1):
            if i != nrow - 1:
                all_edges.append((ncol * i + j, ncol * (i + 1) + j))
            if j != ncol:
                all_edges.append((ncol * i + j, ncol * i + j + 1))
    return np.array(all_edges)


@njit
def calc_distance(q1, q2, nrow, ncol):  # calc_distance_njit in notebook
    # Given two qubits q1, q2 (1-indexed integers) in length x width grid
    # Output l1 distance between q1 and q2 in grid

    grid = np.arange(1, nrow * ncol + 1).reshape((nrow, -1))
    pos1 = np.argwhere(grid == q1)[0]
    pos2 = np.argwhere(grid == q2)[0]

    return np.sum(np.abs(pos1 - pos2))


# without jit but still fastest, v3 in notenook
def get_nearby_qubit_pairs(d, nrow, ncol):
    # 500x faster but order of pairs is slightly different
    # first right-edge than down-edge instead of down-edge then right-edge for each node
    nnodes = nrow*ncol
    xidcs, yidcs = np.unravel_index(range(nnodes), shape=(nrow, ncol))
    node_coords = np.stack((xidcs, yidcs)).T
    dist = cdist(node_coords, node_coords, metric='cityblock').squeeze()
    dist = np.triu(dist)-np.tri(*dist.shape, k=-1)

    ridcs, cidcs = np.where(dist == d)

    return np.stack([ridcs+1, cidcs+1]).T


@njit
def get_local_region_qubits(q, delta1, nrow=5, ncol=5):
    # Given a qubit q (1-indexed integer) in length x width grid and radius delta1
    # delta1 = -1 if all qubits are in local region
    # Output list of qubits (1-indexed integers) within a radius of delta1 of q
    nnodes = nrow * ncol
    if delta1 == 0:
        return [q]
    elif delta1 == -1:
        return list(range(1, nnodes + 1))

    local_qubits = []
    for q2 in range(1, nnodes + 1):
        dist = calc_distance(q, q2, nrow=nrow, ncol=ncol)

        if dist <= delta1:
            local_qubits.append(q2)

    return local_qubits


@njit
def get_local_region_edges(q1, q2, delta1, nrow=5, ncol=5):
    # Given two qubits q1, q2 (1-indexed integers) in length x width grid and radius delta1
    # delta1 = -1 if all qubits are in local region
    # Output list of tuples of qubits (1-indexed integers) corresponding to edges in local region of radius delta1
    all_edges = get_all_edges(nrow, ncol)

    if delta1 == 0:
        return np.array([(q1, q2)])
    elif delta1 == -1:
        return all_edges

    local_qubits = list(set(get_local_region_qubits(
        q1, delta1, nrow=nrow, ncol=ncol) + get_local_region_qubits(q2, delta1, nrow=nrow, ncol=ncol)))

    local_edges = []
    for src, dst in all_edges:
        if src in local_qubits and dst in local_qubits:
            local_edges.append((src, dst))

    return np.array(local_edges)


# @njit
# def get_local_region_params(q1, q2, delta1, data, ii, nrow=5, ncol=5):  # v3 in notebook
#     # Given two qubits q1, q2 (1-indexed integers) in length x width grid, radius delta1, and input data (i.e., Xfull)
#     # delta1 = -1 if all qubits are considered nearby
#     # Output data but only for parameters corresponding to edges within radius delta1

#     edges = get_local_region_edges(q1, q2, delta1, nrow=nrow, ncol=ncol)
#     edges = set([(src, dst) for src, dst in list(edges)])

#     all_edges = get_all_edges(nrow, ncol)
#     indices = []
#     for idx, e in enumerate(all_edges):
#         if (e[0], e[1]) in edges:
#             indices.append(idx)

#     return np.array([data[ii][j] for j in sorted(indices)])


def get_omega(nrow, ncol, delta1, max_R=1000, seed=0):
    rng = np.random.default_rng(seed=seed)
    omega = []

    all_edges = get_all_edges(nrow, ncol)
    for (q1, q2) in all_edges:
        num_local = get_local_region_edges(
            q1, q2, delta1, nrow=nrow, ncol=ncol).shape[0]
        omega_sub = rng.normal(0, 1, (max_R, num_local))

        omega.append(omega_sub)
    return omega


@njit
def get_local_region_params_vectorized(q1, q2, delta1, data, nrow, ncol):

    # Given two qubits q1, q2 (1-indexed integers) in length x width grid, radius delta1, and input data (i.e., Xfull)
    # delta1 = -1 if all qubits are considered nearby
    # Output data but only for parameters corresponding to edges within radius delta1

    edges = get_local_region_edges(q1, q2, delta1, nrow=nrow, ncol=ncol)
    edges = set([(src, dst) for src, dst in list(edges)])

    all_edges = get_all_edges(nrow, ncol)
    indices = []
    for idx, e in enumerate(all_edges):
        if (e[0], e[1]) in edges:
            indices.append(idx)

    idcs = np.array(sorted(indices))
    return data[:, idcs]


# @njit
# v2 in improved notebook
def get_feature_vectors(delta1, R, data, omega, gamma=1.0, nrow=5, ncol=5):
    # Given radius delta1 and hyperparameter R (number of nonlinear features per local region), input data, and fixed randomness omega
    # delta1 = -1 if all qubits are considered nearby
    # Output concatenated feature vectors
    all_edges = get_all_edges(nrow, ncol)

    # to store all concatenated feature vectors
    all_feature_vectors = np.empty((len(all_edges), len(data) * 2 * R))

    # restrict omega to relevant part:
    omega = [w[:R, :] for w in omega]

    for k, (q1, q2) in enumerate(all_edges):

        data_local = get_local_region_params_vectorized(
            q1, q2, delta1, data, nrow=nrow, ncol=ncol)  # nsamples x num_local_edges
        nsamples, m_local = data_local.shape

        # do nonlinear feature map on each vector in data_local
        # shape R x num_local_region_edges
        w_k = np.ascontiguousarray(omega[k])

        val = np.dot(data_local, w_k.T) * gamma / \
            np.sqrt(m_local)  # nsamples x R
        cosv = np.cos(val)
        sinv = np.sin(val)
        feature = np.stack((cosv, sinv))  # shape 2 x nsamples x R
        # shape nsamples x R  x 2
        feature = np.transpose(feature, axes=(1, 2, 0))
        # alternating cos and sin as in original design, shape nsamples x R*2
        feature = feature.copy().reshape(-1)

        all_feature_vectors[k] = feature

    # all this to make the dimensions in order: nsamples x n_edges x R x 2
    # nedges x nsamples*2*2 -> nedges x nsamples x 2*R
    all_feature_vectors = all_feature_vectors.reshape(
        len(all_edges), nsamples, -1)
    all_feature_vectors = np.transpose(
        all_feature_vectors, axes=(1, 0, 2))  # nsamples x nedges x 2*R
    all_feature_vectors = all_feature_vectors.copy().reshape(nsamples, -1)
    # note all_feature_vectors are of size number of data (N) x (2 * R * number of local regions)
    return all_feature_vectors
