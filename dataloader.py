from itertools import product
import os
import numpy as np


def get_couplings(path, row, col, id_, prefix='heisenberg',
                  template='{prefix}_{row}x{col}_id{id_}_couplings.txt'):
    fname = os.path.join(path, template.format(row=row,
                                               col=col,
                                               id_=id_,
                                               prefix=prefix))
    with open(fname, 'r') as f:
        single_hamiltonian = [float(line.strip()) for line in f]
    return single_hamiltonian


def get_energy(path, row, col, id_, prefix='heisenberg',
               template='{prefix}_{row}x{col}_id{id_}_E.txt'):
    fname = os.path.join(path, template.format(row=row,
                                               col=col,
                                               id_=id_,
                                               prefix=prefix))
    with open(fname, 'r') as f:
        energy = float(f.readline().strip())
    return np.array(energy, dtype=float)


def get_svn(path, row, col, id_, prefix='heisenberg',
            template='{prefix}_{row}x{col}_id{id_}_SvN.txt'):
    fname = os.path.join(path, template.format(row=row,
                                               col=col,
                                               id_=id_,
                                               prefix=prefix))
    with open(fname, 'r') as f:
        energy = float(f.readline().strip())
    return np.array(energy, dtype=float)


def get_samples(path, row, col, id_, prefix='heisenberg',
                template='{prefix}_{row}x{col}_id{id_}_samples.txt', return_as_list=False):
    fname = os.path.join(path, template.format(row=row,
                                               col=col,
                                               id_=id_,
                                               prefix=prefix))
    all_samples = []
    with open(fname, 'r') as f:
        for line in f:
            elems = [int(el.strip()) for el in line.split('\t')]
            all_samples.append(np.array(elems, dtype=int).reshape(row, -1))

    # shape: nsamples x row x col
    all_samples = np.stack(all_samples, axis=0)
    if return_as_list:
        num_samples = all_samples.shape[0]
        all_samples = all_samples.reshape(num_samples, -1).tolist()
    return all_samples


def get_correlation(path, row, col, id_, prefix='heisenberg',
                    template='{prefix}_{row}x{col}_id{id_}_{type_}.txt', type_='XX'):
    assert type_ in ['XX', 'YY', 'ZZ'], \
        f'Unknown type {type_} of correlation function'
    fname = os.path.join(path, template.format(row=row,
                                               col=col,
                                               id_=id_,
                                               prefix=prefix,
                                               type_=type_))
    all_samples = []
    with open(fname, 'r') as f:
        for line in f:
            elems = [float(el.strip()) for el in line.split('\t')]
            all_samples.append(elems)

    num_qubits = row * col
    return np.array(all_samples, dtype=float).reshape(num_qubits, -1)


def shadow_alignment(m, n):
    if m > n:
        return shadow_alignment(n, m)  # make sure m is smaller than n

    if m == n:
        return 3  # same basis and outcome
    elif m % 2 == 0 and m == n - 1:
        # measurments (0,1), (2,3), (4,5),
        return -3  # same basis but different outcome
    else:
        return 0


def calc_correlation_from_samples(samples, row, col):

    num_qubits = row * col
    correl_func = []
    grid_iter = product(range(num_qubits), range(num_qubits))
    for i, j in grid_iter:
        if i == j:
            correl_func.append(1.)
        else:
            corr = 0
            for measurement in samples:  # samples shape: bs x row x col
                # measurement shape: row x col
                corr += shadow_alignment(measurement[i], measurement[j])

            correl_func.append(corr / len(samples))

    return np.array(correl_func, dtype=float).reshape(num_qubits, -1)


def get_data(nrow, ncol=5, shadow_size=50, data_name='orig', normalize=True, verbose=True):
    if data_name == 'orig':
        prefix = 'heisenberg'
        path = 'heisenberg_data'
        # for those runs the physical simulation failed (see github issue of original repo)
        row_id_exceptions = {4: [], 5: [], 6: [28, 29, 30], 7: [],
                             8: [53, 54, 55, 56, 57, 58, 59, 60],
                             9: [35, 36, 37, 38, 39, 40, 76, 77, 78, 79, 80]}
    elif data_name == 'new':
        prefix = 'simulation'
        path = f'new_data/data_{nrow}x{ncol}'
    else:
        raise Exception(f"Cant get data. Unknown {data_name=}")

    data_npz = f"all_data_{nrow}x{ncol}_{shadow_size=}.npz"
    data_npz_path = os.path.join(path, data_npz)
    if os.path.exists(data_npz_path):
        loaded = np.load(data_npz_path)
        Ytrain = loaded['Ytrain']
        Yfull = loaded['Yfull']
        Xfull = loaded['Xfull']
    else:
        Ytrain = []
        Yfull = []
        Xfull = []

        id_it = range(1, 101) if data_name == 'orig' else range(1, 501)

        for id_ in id_it:  # <------------- why 301??!
            if data_name == 'orig':
                if id_ in row_id_exceptions[nrow]:
                    continue

            classical_shadow_big = get_samples(
                path, nrow, ncol, id_, prefix=prefix, return_as_list=True)
            classical_shadow = classical_shadow_big[:shadow_size]
            Ytrain.append(calc_correlation_from_samples(
                classical_shadow, nrow, ncol))
            Yfull.append(get_correlation(path, nrow, ncol, id_, prefix=prefix))
            Xfull.append(get_couplings(path, nrow, ncol, id_, prefix=prefix))
        print(f'saving Ytrain, Yfull, Xfull to {os.path.join(path,data_npz)}')
        np.savez_compressed(data_npz_path, Ytrain=Ytrain,
                            Yfull=Yfull, Xfull=Xfull)

    Ytrain = np.array(Ytrain)
    Yfull = np.array(Yfull)
    Xfull = np.array(Xfull)

    if verbose:
        print(f'Loaded {data_name}_data for {nrow}x{ncol}, {shadow_size=}')
        print("number of data (N) * number of params (m) =", Xfull.shape)
        print("number of data (N) * number of pairs =", Yfull.shape)
    if normalize:
        # Normalize Xfull
        xmin = np.amin(Xfull)
        xmax = np.amax(Xfull)

        # normalize so that all entries are between -1 and 1 using min-max feature scaling
        Xfull_norm = -1 + 2 * (Xfull - xmin) / (xmax - xmin)
        return Xfull_norm, Ytrain, Yfull

    return Xfull, Ytrain, Yfull


if __name__ == "__main__":

    # import cProfile

    # cProfile.run(
    #     "get_data(5, ncol=5, shadow_size=500, data_name='orig', normalize=True)",
    #     sort='tottime')
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib

    rows = range(4, 10)
    nshadows = [50, 100, 250, 500, 1000]
    datasets = ['orig', 'new']
    ntot = len(rows) * len(nshadows) * len(datasets)
    with tqdm_joblib(desc="loading data", total=ntot) as progress_bar:
        Parallel(n_jobs=os.cpu_count())(delayed(get_data)(
            nr, ncol=5, shadow_size=ns, data_name=dn, normalize=True)
            for nr, ns, dn in product(rows, nshadows, datasets))
