import os
from argparse import ArgumentParser, BooleanOptionalAction
from itertools import product
from functools import partial
import warnings
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import numba as nb
import celer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning

from utils import get_feature_vectors, get_omega, get_nearby_qubit_pairs, get_all_edges
from dataloader import get_data


def parse_args(return_parser=False, default_algo='new'):
    parser = ArgumentParser()
    parser.add_argument("--algo-type",
                        default=default_algo,
                        choices=['orig', 'new'],
                        type=str,
                        help="type of learning algorithm")
    parser.add_argument("--data-name",
                        default='new',
                        choices=['orig', 'new'],
                        type=str,
                        help="dataset type")
    parser.add_argument("--test-size",
                        default=0.1,
                        choices=[0.1, 0.3, 0.5, 0.7, 0.9],
                        type=float,
                        help="test set fraction")
    parser.add_argument("--shadow-size",
                        default=50,
                        choices=[50, 100, 250, 500, 1000],
                        type=int,
                        help="number of samples to construct shadows")
    parser.add_argument("--qubit-dist",
                        default=1,
                        type=int,
                        help="distance of qubits we care about predicting correlation for")
    parser.add_argument("--nrow",
                        default=4,
                        choices=[4, 5, 6, 7, 8, 9],
                        type=int,
                        help="num of rows in grid")
    parser.add_argument("--ncol",
                        default=5,
                        choices=[5],
                        type=int,
                        help="num of col in grid")
    parser.add_argument("--model-seed",
                        default=42,
                        type=int,
                        help="seed for model")
    parser.add_argument("--data-seed",
                        default=0,
                        type=int,
                        help="seed for data splitting")
    parser.add_argument("--num-cross-val",
                        default=4,
                        type=int,
                        help="number of cross validation folds to do")
    parser.add_argument("--pbar",
                        default=True,
                        type=bool,
                        action=BooleanOptionalAction,
                        help="display tqdm-progress bar")
    parser.add_argument("--debug",
                        default=False,
                        type=bool,
                        action=BooleanOptionalAction,
                        help="do debug run")

    args, unknown_strgs = parser.parse_known_args()
    if args.algo_type == 'new':
        # args for only new algo
        parser.add_argument("--delta1",
                            default=0,
                            type=int,
                            help="distance parameter for local region")
        parser.add_argument("--max-R",
                            default=1000,
                            type=int,
                            help="parameter for omega features")
        parser.add_argument("--omega-seed",
                            default=0,
                            type=int,
                            help="seed for omega")
        parser.add_argument("--lasso-lib",
                            default='sklearn',
                            choices=['sklearn', 'celer'],
                            type=str,
                            help="which lasso library to use")
        parser.add_argument("--lasso-tol",
                            default=1e-3,
                            type=float,
                            help="tolerance to use for lasso")
        parser.add_argument("--lasso-maxiter",
                            default=10000,
                            type=float,
                            help="max iter to use for lasso")
        parser.add_argument("--lasso-maxep",
                            default=50000,
                            type=float,
                            help="max epoch to use for celer-lasso")
    elif args.algo_type == 'orig':
        parser.add_argument("--dirichlet-include-x",
                            default=True,
                            type=bool,
                            action=BooleanOptionalAction,
                            help="include x into dirichlet kernel")
        parser.add_argument("--dirichlet-size-factor",
                            default=4,
                            type=int,
                            help="size factor for dirichlet kernel")
        parser.add_argument("--ntk-normalization",
                            default=False,
                            type=bool,
                            action=BooleanOptionalAction,
                            help="do normalization for ntk kernels")
        parser.add_argument("--ntk-hidden-dim",
                            default=32,
                            type=int,
                            help="hidden dim of the ntk")
        parser.add_argument("--svr-tol",
                            default=1e-3,
                            type=float,
                            help="tolerance to use for svr")

    if return_parser:
        return parser
    args = parser.parse_args()
    return args


def celer_ml_method(Cx, maxep=50000, maxiter=10000, tol=1e-4):
    return celer.Lasso(
        alpha=Cx, max_epochs=maxep, max_iter=maxiter, tol=tol)


def sklearn_ml_method(Cx, maxiter=10000, tol=1e-4):
    return linear_model.Lasso(alpha=Cx, max_iter=maxiter, tol=tol)
    # sklearn max_iter was 10k in all runs sent in email


def train_and_predict(q1, q2, hp, data, omega):
    # consider the pair (q1, q2)
    data = deepcopy(data)
    omega = [deepcopy(w) for w in omega]

    Xfull, Ytrain, Yfull = data
    # Xfull shape nsamples x num edges
    # Ytrain shape nsamples x nnodes x nnodes
    # Yfull shape nsamples x nnodes x nnodes
    if hp.algo_type == 'new':
        Xfull_norm = Xfull

    nnodes = hp.nrow * hp.ncol
    nsamples = len(Xfull)

    # training data (estimated from measurement data)
    y = Ytrain[:, q1 - 1, q2 - 1]
    X_train, X_test, y_train, y_test = train_test_split(
        Xfull_norm, y, test_size=hp.test_size, random_state=hp.data_seed)

    # testing data (exact expectation values)
    y_clean = Yfull[:, q1 - 1, q2 - 1]

    _, _, _, y_test_clean = train_test_split(
        Xfull_norm, y_clean, test_size=hp.test_size, random_state=hp.data_seed)

    # use cross validation to find the best hyperparameters
    best_cv_score, test_score = 999.0, 999.0
    if hp.lasso_lib == 'sklearn':
        ML_method = partial(sklearn_ml_method,
                            maxiter=hp.lasso_maxiter, tol=hp.lasso_tol)
    else:
        ML_method = partial(celer_ml_method,
                            maxep=hp.lasso_maxep, maxiter=hp.lasso_maxiter, tol=hp.lasso_tol)

    best_coef = []

    # to have edges in non numpy format
    all_edges = get_all_edges(hp.nrow, hp.ncol).tolist()
    R_it = [5, 10, 20, 40]
    gamma_it = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75]
    np.random.seed(hp.model_seed)
    for R, gamma in product(R_it, gamma_it):
        # feature mapping
        Xfeature_train = get_feature_vectors(
            hp.delta1, R, X_train, omega, gamma, nrow=hp.nrow, ncol=hp.ncol)
        Xfeature_test = get_feature_vectors(
            hp.delta1, R, X_test, omega, gamma, nrow=hp.nrow, ncol=hp.ncol)

        # sklearn docs: To avoid unnecessary memory duplication
        # the X argument of the fit method should be directly passed as a Fortran-contiguous numpy array.
        Xfeature_train = np.asfortranarray(Xfeature_train)
        Xfeature_test = np.asfortranarray(Xfeature_test)

        for alpha in [2**(-8), 2**(-7), 2**(-6), 2**(-5)]:
            score = -np.mean(cross_val_score(ML_method(alpha), Xfeature_train, y_train,
                                             cv=hp.num_cross_val,
                                             scoring="neg_root_mean_squared_error"))
            if best_cv_score > score:
                clf = ML_method(alpha).fit(Xfeature_train, y_train.ravel())
                test_score = np.linalg.norm(clf.predict(Xfeature_test).ravel() -
                                            y_test_clean.ravel()) / (len(y_test) ** 0.5)
                best_cv_score = score
                best_coef = clf.coef_.reshape(len(all_edges), 2 * R)

                # coef = clf.coef_.reshape((len(all_edges), 2 * R))
                # print(list(zip(all_edges, np.linalg.norm(coef, axis=1))))
                # print(R, gamma, alpha, score, test_score)
    coef_edges = list(zip(all_edges, np.linalg.norm(best_coef, axis=1)))
    return best_cv_score, test_score, coef_edges


def prepare_path(path):
    if os.path.exists(path):
        return

    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)

    # if path is only folder
    if os.path.splitext(path)[-1] == '':
        os.makedirs(path, exist_ok=True)
    return


def main(hp):
    assert hp.algo_type == 'new', 'This function should only be used to train new algo'
    print(f"received following hparams: ")
    print(hp)
    if hp.debug:
        result_dir = 'debug'
    else:
        result_dir = 'clean_results'

    if hp.qubit_dist == -1:
        qubits_suffix = f"all_qubits"
    else:
        qubits_suffix = f"qubits_d={hp.qubit_dist}"

    if hp.data_name == 'new':
        newdata_suffix = "500data"
    else:
        newdata_suffix = ""

    if hp.lasso_lib == 'celer':
        data_path = './{}/{}_algorithm_{}_maxiter={}_maxep={}_tol={}_seed={}/test_size={}_shadow_size={}_{}'.format(
            result_dir, hp.algo_type, hp.lasso_lib, hp.lasso_maxiter, hp.lasso_maxep, hp.lasso_tol, hp.data_seed, hp.test_size, hp.shadow_size, qubits_suffix)
    else:
        data_path = './{}/{}_algorithm_{}_maxiter={}_tol={}_seed={}/test_size={}_shadow_size={}_{}'.format(
            result_dir, hp.algo_type, hp.lasso_lib, hp.lasso_maxiter, hp.lasso_tol, hp.data_seed, hp.test_size, hp.shadow_size, qubits_suffix)

    if newdata_suffix:
        data_path = data_path.replace(
            'algorithm_', f"algorithm_{newdata_suffix}_")

    # get data and omega/kernel
    data = get_data(nrow=hp.nrow, ncol=hp.ncol, shadow_size=hp.shadow_size,
                    data_name=hp.data_name, normalize=True)
    # data = Xfull, Ytrain, Yfull
    # or in normalized case Xfull_normalized instead of Xfull
    omega = get_omega(hp.nrow, hp.ncol, hp.delta1,
                      max_R=hp.max_R, seed=hp.omega_seed)
    omega = nb.typed.List(omega)

    result_path = '{}/results_{}x{}_{}_data.txt'.format(
        data_path, hp.nrow, hp.ncol, hp.data_name)
    coef_path = '{}/coefficients_{}x{}_{}_data.txt'.format(
        data_path, hp.nrow, hp.ncol, hp.data_name)

    for path in [data_path, result_path, coef_path]:
        prepare_path(path)

    warning_msgs = dict()
    with open(result_path, 'w') as f1, open(coef_path, 'w') as f2:
        print(f'Writing result to {result_path}')
        print(f'Writing coefficients to {coef_path}')
        qubits = get_nearby_qubit_pairs(hp.qubit_dist, hp.nrow, hp.ncol)

        qbit_iter = tqdm(qubits) if hp.pbar else qubits

        for (q1, q2) in qbit_iter:
            if hp.pbar:
                print('(q1, q2) =', (q1, q2))
            print('(q1, q2) =', (q1, q2), file=f1)
            print('(q1, q2) =', (q1, q2), file=f2)
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                # this can trigger sklearn.ConvergenceWarning
                res = train_and_predict(q1, q2, hp, data, omega)

                # warning handling:
                # ignore any non-custom warnings that may be in the list
                nonConvergenceWarnings = []
                other_warnings = []
                for wi in w:
                    if issubclass(wi.category, ConvergenceWarning):
                        nonConvergenceWarnings.append(wi)
                    else:
                        other_warnings.append(wi)
                # w = list(filter(lambda i: issubclass(i.category, ConvergenceWarning), w))

                for wi in other_warnings:
                    print(wi.message)

                if len(nonConvergenceWarnings):
                    # do something with the warning
                    print(f'Warning occured for edge ({q1}, {q2})')
                    print(nonConvergenceWarnings[0].message)
                    warning_msgs[(q1, q2)] = True

            # print(res)
            print(res[0:2], file=f1)
            print(res[2], file=f2)

    if warning_msgs:
        warnings_path = '{}/warnings_{}x{}_{}_data.txt'.format(
            data_path, hp.nrow, hp.ncol, hp.data_name)
        prepare_path(warnings_path)

        with open(warnings_path, 'w') as f:
            for (q1, q2) in warning_msgs.keys():
                print('(q1, q2) =', (q1, q2), file=f)

    return hp, warning_msgs.keys(), len(warning_msgs)


if __name__ == "__main__":
    args = parse_args()

    if args.algo_type == 'orig':
        raise RuntimeError('This script is not for Original kernel methods.')

    main(args)
