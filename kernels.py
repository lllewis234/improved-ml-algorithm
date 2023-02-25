import os
import numpy as np
from numba import njit
from itertools import product
from neural_tangents import stax

from dataloader import get_data

# Neural tangent kernel
# Traditional ML methods and techniques


@njit
def create_dirichlet_kernel(Xfull, size_factor=4, include_x=True):
    assert isinstance(size_factor, int), 'size_factor needs to be integer'
    if include_x:
        assert size_factor % 2 == 0, ('If x should be included into kernel'
                                      ' we need an additional dimension, thus size_factor needs to be even')

    if include_x:
        sf2 = size_factor // 2 - 1
    else:
        sf2 = (size_factor - 1) // 2

    kernel_dir = np.zeros((len(Xfull), Xfull.shape[1] * size_factor))
    for i, x1 in enumerate(Xfull):
        cnt = 0
        for k in range(len(x1)):
            if include_x:
                kernel_dir[i, cnt] = x1[k]
                cnt += 1

            for k1 in range(-sf2, sf2 + 1):
                kernel_dir[i, cnt] = np.cos(np.pi * k1 * x1[k])
                cnt += 1
    print("constructed Dirichlet kernel")
    return kernel_dir


def create_ntk_kernel(Xfull, data_name, nrow, ncol, shadow_size, nlayer=2, hidden_dim=32,
                      do_normalization=False, force_recompute=False):
    if data_name == 'orig':
        path = 'heisenberg_data'
        data_suffix = ""
    elif data_name == 'new':
        path = f'new_data/data_{nrow}x{ncol}'
        data_suffix = "500"
        assert Xfull.shape[0] == 500, "new data is only for 500 samples"
    else:
        raise Exception(f"Cant get kernels. Unknown {data_name=}")

    data_npy = f"ntk{nlayer}h{hidden_dim}_{nrow}x{ncol}_{shadow_size=}_norm{do_normalization}_{data_suffix}.npy"
    data_npy_path = os.path.join(path, data_npy)
    if os.path.exists(data_npy_path) and not force_recompute:
        kernel = np.load(data_npy_path)
    else:
        #
        # Neural tangent kernel
        #
        hidden_block = []
        for _ in range(nlayer):
            hidden_block.append(stax.Dense(hidden_dim))
            hidden_block.append(stax.Relu())

        init_fn, apply_fn, kernel_fn = stax.serial(
            *hidden_block,
            stax.Dense(1)
        )
        kernel = kernel_fn(Xfull, Xfull, 'ntk')
        kernel = np.asarray(kernel).copy()

        np.save(data_npy_path, kernel)
        print(f"constructed neural tangent kernel for {data_name} data, "
              f"{nrow}x{ncol}, {shadow_size=}. {nlayer=}, {hidden_dim=}")

    if do_normalization:
        # correct approach to normalize to correl_coef matrix
        kernel_norm = kernel.copy()
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                kernel_norm[i, j] /= (kernel[i, i] * kernel[j, j])**0.5

        return kernel_norm
    else:
        return kernel


def get_data_and_create_ntk(nrow, ncol, shadow_size, data_name='orig', nlayer=2, hidden_dim=32, do_normalization=False):
    Xfull, _, _ = get_data(nrow, ncol=ncol, shadow_size=shadow_size,
                           data_name=data_name, normalize=False)
    create_ntk_kernel(Xfull, data_name=data_name, nrow=nrow, ncol=ncol, shadow_size=shadow_size,
                      nlayer=nlayer, hidden_dim=hidden_dim, do_normalization=do_normalization)
    return


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    rows = range(4, 10)
    nshadows = [50, 100, 250, 500, 1000]
    datasets = ['orig', 'new']
    nlayers = range(2, 6)
    norm = [True, False]
    ntot = len(rows) * len(nshadows) * len(datasets) * \
        len(nlayers) * len(norm)
    with tqdm_joblib(desc="loading data", total=ntot) as progress_bar:
        Parallel(n_jobs=16)(delayed(get_data_and_create_ntk)(
            nr, ncol=5, shadow_size=ns, data_name=dn, nlayer=nl, do_normalization=n)
            for nr, ns, dn, nl, n, nd in product(rows, nshadows, datasets, nlayers, norm))
