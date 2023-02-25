# import before all to avoid gpu usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# main modules
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.exceptions import ConvergenceWarning
import numpy as np

# additional modules
import warnings
from copy import deepcopy
from tqdm import tqdm

# custom modules
from kernels import create_dirichlet_kernel, create_ntk_kernel
from dataloader import get_data
from utils import get_nearby_qubit_pairs
from train import prepare_path, parse_args
#from sklearn.ensemble import RandomForestRegressor
#import jax


# opt="linear" or "rbf"
def train_and_predict(q1, q2, hp, data, kernel, opt="linear", method=""):
    data = deepcopy(data)
    kernel = kernel.copy()
    Xfull, Ytrain, Yfull = data

    # instance norm only helps old dirichlet
    # if method == "dirichlet": # skip for gauss and ntk
    #     # kernel shape nsamples: feature dim
    #     # instance-wise normalization
    #     instance_norms = np.linalg.norm(kernel, axis=-1, keepdims=True)
    #     kernel /= instance_norms

    # training data (estimated from measurement data)
    y = Ytrain[:, q1 - 1, q2 - 1]
    # y = np.array([Ytrain[i][k] for i in range(len(Xfull))])

    X_train, X_test, y_train, y_test = train_test_split(
        kernel, y, test_size=hp.test_size, random_state=hp.data_seed)

    # testing data (exact expectation values)
    y_clean = Yfull[:, q1 - 1, q2 - 1]
    # y_clean = np.array([Yfull[i][k] for i in range(len(Xfull))])
    _, _, _, y_test_clean = train_test_split(
        kernel, y_clean, test_size=hp.test_size, random_state=hp.data_seed)

    # use cross validation to find the best method + hyper-param
    best_cv_score, test_score = 999.0, 999.0
    for ML_method in [(lambda Cx: svm.SVR(kernel=opt, C=Cx, tol=hp.svr_tol)), (lambda Cx: KernelRidge(kernel=opt, alpha=1 / (2 * Cx)))]:
        for C in [0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0]:
            score = -np.mean(cross_val_score(ML_method(C), X_train, y_train,
                                             cv=hp.num_cross_val,
                                             scoring="neg_root_mean_squared_error"))
            if best_cv_score > score:
                clf = ML_method(C).fit(X_train, y_train.ravel())
                test_score = np.linalg.norm(clf.predict(X_test).ravel() -
                                            y_test_clean.ravel()) / (len(y_test) ** 0.5)
                best_cv_score = score

    return best_cv_score, test_score


def main(hp):
    assert hp.algo_type == 'orig', 'This function should only be used to train orig algo'

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

    data_path = './{}/{}_algorithm_svrtol={}_ntk-norm={}_diri-inclx={}_diri-sf={}/test_size={}_shadow_size={}_{}'.format(
        result_dir, hp.algo_type, hp.svr_tol, hp.ntk_normalization, hp.dirichlet_include_x, hp.dirichlet_size_factor,
        hp.test_size, hp.shadow_size, qubits_suffix)

    if newdata_suffix:
        data_path = data_path.replace(
            'algorithm_', f"algorithm_{newdata_suffix}_")

    # get data and omega/kernel
    data = get_data(nrow=hp.nrow, ncol=hp.ncol, shadow_size=hp.shadow_size,
                    data_name=hp.data_name, normalize=False)  # tuple Xfull, Ytrain, Yfull
    # create kernels
    kernel_dir = create_dirichlet_kernel(
        data[0], hp.dirichlet_size_factor, hp.dirichlet_include_x)
    list_kernel_NN = [create_ntk_kernel(data[0], hp.data_name, hp.nrow, hp.ncol, hp.shadow_size,
                                        nlayer=nlayer, hidden_dim=hp.ntk_hidden_dim,
                                        do_normalization=hp.ntk_normalization) for nlayer in range(2, 6)]

    result_path = '{}/results_{}x{}_{}_data.txt'.format(
        data_path, hp.nrow, hp.ncol, hp.data_name)

    for path in [data_path, result_path]:
        prepare_path(path)

    warning_msgs = dict()
    with open(result_path, 'w') as f1:
        print(f'Writing result to {result_path}')
        qubits = get_nearby_qubit_pairs(hp.qubit_dist, hp.nrow, hp.ncol)

        qbit_iter = tqdm(qubits) if hp.pbar else qubits

        for (q1, q2) in qbit_iter:
            if hp.pbar:
                print('(q1, q2) =', (q1, q2))
            print('(q1, q2) =', (q1, q2), file=f1)
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                # Dirichlet
                res = train_and_predict(
                    q1, q2, hp, data, kernel_dir, method='dirichlet')
                print("Dirich. kernel", res, file=f1)

                # RBF
                res = train_and_predict(
                    q1, q2, hp, data, data[0], opt="rbf", method='gauss')  # data[0] is Xfull
                print("Gaussi. kernel", res, file=f1)

                # Neural tangent
                for kernel_NN in list_kernel_NN:
                    res = train_and_predict(
                        q1, q2, hp, data, kernel_NN, method='ntk')
                    print("Neur. T kernel", res, file=f1)

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
    if warning_msgs:
        warnings_path = '{}/warnings_{}x{}_{}_data.txt'.format(
            data_path, hp.nrow, hp.ncol, hp.data_name)
        prepare_path(warnings_path)

        with open(warnings_path, 'w') as f:
            for (q1, q2) in warning_msgs.keys():
                print('(q1, q2) =', (q1, q2), file=f)

    return hp, warning_msgs.keys(), len(warning_msgs)


if __name__ == "__main__":
    args = parse_args(default_algo='orig')

    if args.algo_type == 'new':
        raise RuntimeError('This script is not for new kernel methods.')

    main(args)
