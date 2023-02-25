import os
from collections import defaultdict

from math import floor
from glob import glob
from parse import parse
import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

def extract(row):
    """Extracts the parameters from the path and returns a row with the parameters"""
    if 'new_algorithm' in row.path:
        row = extract_new(row)
    else:
        row = extract_orig(row)
    
    row['algo_setting'] = row.path.split(os.sep)[2]
    return row

def extract_new(row):
    if 'celer' in row.path:
        pat = "{}new_algorithm_{lassolib}_maxiter={maxiter:d}_maxep={maxep:d}_tol={tol:f}/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    elif 'sklearn' in row.path:
        pat = "{}new_algorithm_{lassolib}_maxiter={maxiter:d}_tol={tol:f}/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    
    
    temp = parse(pat, row.path)
    temp = pd.Series(temp.named)
    temp['path'] = row.path
    temp['algo'] = 'new'
    return temp
    
def extract_orig(row):
    pat0 = "{}orig_algorithm_svrtol={srvtol:f}_ntk-norm={ntknorm}_diri-inclx={dirinclx}_diri-sf={dirsf:d}/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    pat1 = "{}orig_algorithm_{algo_suffix}_svrtol={srvtol:f}_ntk-norm={ntknorm}_diri-inclx={dirinclx}_diri-sf={dirsf:d}/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    
    # try only orig_algorithm since this is output of unmodified orig notebook
    pat2 = "{}orig_algorithm_processed/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    pat3 = "{}orig_algorithm_processed_{algo_suffix}/test_size={testsize:f}_shadow_size={shadowsize:d}_qubits_d={qubitdist:d}/results_{nrow:d}x{ncol:d}_{dataname}_data.txt"
    
    patterns = [pat0, pat1, pat2, pat3]
    for pat in patterns:
        temp = parse(pat, row.path)
        if temp:
            temp = pd.Series(temp.named)
            break

    if temp is None:
        raise RuntimeError(f"unknown path pattern {row.path}")

            
    temp['path'] = row.path
    temp['algo'] = 'orig'
    return temp

def read_results(row):
    """Reads the results from the file and returns a row with the results"""
    if row['algo'] == 'new':
        return read_results_new(row)
    elif row['algo'] == 'orig':
        return read_results_orig(row)

def read_results_new(row):
    
    pat1 = "(q1, q2) = ({:d}, {:d})"
    pat2 = "({:g}, {:g})"
    
    edges = []
    best_cv_score = []
    test_score = []
    with open(row.path, 'r') as f:
        try:
            for lnum, line in enumerate(f):
                line = line.strip()
                edge = parse(pat1, line).fixed
                edges.append(edge)

                line2 = next(f).strip()
                cv, test = parse(pat2, line2).fixed

                best_cv_score.append(cv)
                test_score.append(test)
        except Exception as e:
            print(f"Parsing error occured at line {lnum}, row number:{row.name} {row.path}")
            print(line)
            raise
    
    row['nedges'] = len(edges)
    row['edges'] = edges
    row['best_cv_score_new'] = best_cv_score
    row['test_score_new'] = test_score
    return row

def read_results_orig(row):
    
    pat1 = "(q1, q2) = ({:d}, {:d})"
    pat_dirichlet = "Dirich. kernel ({:g}, {:g})" # general number format , (either d, f or e)
    pat_gauss = "Gaussi. kernel ({:g}, {:g})"
    pat_ntk = "Neur. T kernel ({:g}, {:g})"
    
    ntklayer = range(2,6)
    
    edges = []
    best_cv_score = defaultdict(list)
    test_score = defaultdict(list)
    
    with open(row.path, 'r') as f:
        try:
            lnum = -1
            for cnt, line in enumerate(f):
                lnum += 1
                line = line.strip()
                edge = parse(pat1, line).fixed
                edges.append(edge)

                # dirichlet
                line_dirichlet = next(f).strip()
                lnum += 1
                cv, test = parse(pat_dirichlet, line_dirichlet).fixed
                best_cv_score['dirichlet'].append(cv)
                test_score['dirichlet'].append(test)

                # gauss
                line_gauss = next(f).strip()
                lnum += 1
                cv, test = parse(pat_gauss, line_gauss).fixed
                best_cv_score['gauss'].append(cv)
                test_score['gauss'].append(test)

                #ntk
                for k in ntklayer:
                    line_ntk = next(f).strip()
                    lnum += 1
                    cv, test = parse(pat_ntk, line_ntk).fixed
                    best_cv_score[f'ntk{k}'].append(cv)
                    test_score[f'ntk{k}'].append(test)
        except Exception as e:
            print(f"Parsing error occured at line {lnum}, row number:{row.name} {row.path}")
            print(line)
                  
                  
    
    row['nedges'] = len(edges)
    row['edges'] = edges
    row['best_cv_score_dirichlet'] = best_cv_score['dirichlet']
    row['test_score_dirichlet'] = test_score['dirichlet']
    
    row['best_cv_score_gauss'] = best_cv_score['gauss']
    row['test_score_gauss'] = test_score['gauss']
    
    for k in ntklayer:
        row[f'best_cv_score_ntk{k}'] = best_cv_score[f'ntk{k}']
        row[f'test_score_ntk{k}'] = test_score[f'ntk{k}']
    
    return row
    
def calc_nsamples(row):
    if row['dataname'] == 'new':
        if '300' in row['path'].split(os.sep)[0]:
            tot_samples = 300
        else:
            tot_samples = 500
    else:
        if row['nrow'] in [4,5,7]:
            tot_samples = 100
        elif row['nrow'] == 6:
            tot_samples = 97
        elif row['nrow'] == 8:
            tot_samples = 92
        elif row['nrow'] == 9:
            tot_samples = 89
        
    row['nsamples'] = row['testsize'] * tot_samples
    return row

def create_matrix(row):
    nrow = row['nrow']
    nnodes = nrow * 5
    train_arr = np.zeros((nnodes, nnodes))
    test_arr = np.zeros((nnodes, nnodes))
    for (src, dst), train, test in zip(row['edges'], row['best_cv_score'], row['test_score']):
        train_arr[src-1, dst-1] = train
        test_arr[src-1, dst-1] = test
    
    row['train_mat'] = train_arr
    row['test_mat'] = test_arr
    return row

def calc_avg(row):
    cols = row.keys()
    
    # mean over edges
    for c in cols:
        if c.startswith('best_cv_score'):
            avgc = c.replace('best_cv_score', 'avg_train')
            row[avgc] = np.mean(row[c])
        elif c.startswith('test_score'):
            avgc = c.replace('test_score', 'avg_test')
            row[avgc] = np.mean(row[c])
        else:
            continue
    
    cols = row.keys()
    orig_methods_train = [c for c in cols if 'avg_train' in c and 'new' not in c]
    orig_methods_test = [c for c in cols if 'avg_test' in c and 'new' not in c]
    row['avg_test_orig'] = np.min(row[orig_methods_test])
    row['avg_train_orig'] = np.min(row[orig_methods_train])
    return row

def get_results(globpath):
    files = glob(os.path.join(globpath,'**/results*.txt'), recursive=True)

    df = pd.DataFrame({'path':files})
    print(f"Found {len(df)} files")
    df = df.progress_apply(extract, axis=1)
    print(f"Extracted parameters from {len(df)} paths")
    df = df.progress_apply(read_results, axis=1)
    print(f"Read in {len(df)} files")
    df = df.apply(calc_nsamples, axis=1)
    df = df.apply(calc_avg, axis=1)
    
    return df

def convert_df(df):
    df = df.copy()
    # drop additional metrics 
    df = df.drop(columns=[c for c in df.columns if 'best' in c])
    df = df.drop(columns=[c for c in df.columns if 'test_score' in c])
    df = df.drop(columns=[c for c in df.columns if 'avg_train' in c])


    # drop columns which dont vary of if there are two values and one of them is nan then also drop
    # this is the case for parameters columns of orig-algo which dont vary or 
    # parameter columns of new-algo  which dont vary
    todrop = ['edges', 'path', 'nedges', 'algo_setting']
    whitelist = ['qubitdist', 'nrow', 'dataname', 'testsize', 'nsamples', 'nrow']
    for c in df.columns:
        try:
            if c in whitelist:
                continue # skip
            if len(df[c].unique()) == 1:
                todrop.append(c)
            elif len(df[c].unique()) == 2:
                # and one is nan , then also delete
                if pd.isnull(df[c].unique()).any():
                    todrop.append(c)
        except:
            continue
    print(f"dropping {len(todrop)} columns: {todrop}")
    df2 = df.drop(columns=todrop)

    # we need to handle the orig-algo rows differently since they contain the entries for columns 
    # avg_test_dirichlet 	avg_test_gauss 	avg_test_ntk2 	avg_test_ntk3 	avg_test_ntk4 	avg_test_ntk5 	avg_test_orig
    # avg_test_orig is min of [avg_test_dirichlet,avg_test_gauss,avg_test_ntk2,avg_test_ntk3,avg_test_ntk4,avg_test_ntk5]
    df_orig = df2[df2['algo'] == 'orig']
    df_orig = df_orig.dropna(axis=1) # drop avg_test_bew col

    # convert table format
    # instead of storing metrics in each col seperately we merge them into one "avg_test" col 
    # and add a second col which indicates the method: dirichlet/gauss/ntk2,3,4,5
    df_orig2 = pd.wide_to_long(df_orig, stubnames=['avg_test_'], i=['testsize', 'shadowsize', 'qubitdist', 'nrow', 'dataname', 'nsamples'], j='method', suffix=".+")
    df_orig2 = df_orig2.rename({'avg_test_':'avg_test'}, axis=1)
    df_orig2 = df_orig2.reset_index()

    # the new alogs only have entries in the col avg_test_new
    df_new = df2[df2['algo'] == 'new']
    df_new = df_new.dropna(axis=1) # drop other metric cols

    df_new2 = df_new.rename({'avg_test_new':'avg_test'}, axis=1)
    df_new2['method'] = 'new'

    # append both tables with common structure
    df3 = df_new2.append(df_orig2) 
    df3.head()

    print("unique values per column: ")
    for c in df3.columns:
        print(f"column {c}:\tunique values: {df3[c].unique()}")

    return df3

def calc_trainsize(row, ntotal=500):
    # test_size is rounded to next biggest int from sklearn train_test_split
    # thus train is rounded down
    if row.dataname == 'orig':
        if row['nrow'] in [4,5,7]:
            ntotal = 100
        elif row['nrow'] == 6:
            ntotal = 97
        elif row['nrow'] == 8:
            ntotal = 92
        elif row['nrow'] == 9:
            ntotal = 89
        return floor(ntotal - row['nsamples'])
    if row.dataname == 'new':
        return floor(ntotal - row['nsamples'])