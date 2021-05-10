import pandas as pd
import pickle as pkl
import os
from utils_ph import *
import numpy as np

def compute_df(mu_0, mu_1, lam_0,lam_1, path_before, path_after, ub_v):




    df1 = pkl.load(open(path_before, 'rb'))

    df1.loc[df1['c'] == 0, 'mu0'] = df1.loc[df1['c'] == 0, 'v']
    df1.loc[df1['c'] > 0, 'mu0'] = df1.loc[df1['c'] > 0, 'v'] + 1 - df1.loc[df1['c'] > 0, 'Ar']
    df1.loc[df1['c'] == 0, 'lam0lam1'] = 0
    df1.loc[df1['c'] == 0, 'lam0lam1mu0'] = 0
    df1.loc[df1['c'] > 0, 'lam0lam1'] = df1.loc[df1['c'] > 0, 'Id']
    df1.loc[df1['c'] > 0, 'lam0lam1mu0'] = df1.loc[df1['c'] > 0, 'c'] - df1.loc[df1['c'] > 0, 'Id'] + df1.loc[
        df1['c'] > 0, 'Ar'] - 1
    df1.loc[df1['c'] == 0, 'l1'] = 0
    df1.loc[df1['c'] == 0, 'l2'] = 0
    df1.loc[df1['c'] > 0, 'l1'] = df1.loc[df1['c'] > 0, 'Ar'] - 1
    df1.loc[df1['c'] > 0, 'l2'] = df1.loc[df1['c'] > 0, 'c'] - df1.loc[df1['c'] > 0, 'Id']

    df1['prob_c'] = df1.apply(lambda x: marg_prob_c(x.v, x.c, mu_0, lam_0, lam_1, mu_1), axis=1)
    df1['prob_v'] = df1.v.apply(lambda x: prob_v(lam_0, lam_1, x))
    df1['prob_arrival'] = df1.apply(lambda x: prob_arrival(x.l1, x.l2, lam_0, lam_1, mu_0, x.number), axis=1)
    df1['total_prob'] = df1['prob_c'] * df1['prob_v'] * df1['prob_arrival']

    df1['event'] = df1.mu0.apply(lambda x: str(int(x)) + '_') + df1.lam0lam1.apply(
        lambda x: str(int(x)) + '_') + df1.lam0lam1mu0.apply(lambda x: str(int(x)))
    df_merged = merge_cases(df1)
    df_merged['prob'] = df_merged['prob'].astype('float')
    df_rates = df_merged['event'].str.split('_', expand=True)
    df_rates = df_rates.rename(columns={0: "mu0", 1: "lam0lam1", 2: 'lam0lam1mu0'})
    result = pd.concat([df_merged, df_rates], axis=1)

    result['mu0'] = result['mu0'].astype(int)
    result['lam0lam1'] = result['lam0lam1'].astype(int)
    result['lam0lam1mu0'] = result['lam0lam1mu0'].astype(int)

    mu0_avg_max_v = round(df1.loc[df1['v'] == ub_v-1, 'mu0'].mean()) + 1
    lam0lam1_avg_max_v = round(df1.loc[df1['v'] == ub_v-1, 'lam0lam1'].mean()) + 1
    lam0lam1mu0_avg_max_v = round(df1.loc[df1['v'] == ub_v-1, 'lam0lam1mu0'].mean()) + 1

    pkl.dump((mu0_avg_max_v, lam0lam1_avg_max_v, lam0lam1mu0_avg_max_v), open('mean_num_rates_ub_v.pkl', 'wb'))




    pkl.dump(result, open(path_after, 'wb'))

    # return result