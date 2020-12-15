import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


def avg_waiting_time(p0, mu, lamb=1.0):
    rho = (p0 / mu[0] + (1 - p0) / mu[1]) * lamb
    expected_square = (2 * p0) / mu[0] ** 2 + (2 * (1 - p0)) / mu[1] ** 2
    expected = p0 / mu[0] + (1 - p0) / mu[1]
    avg_waiting = expected + lamb * expected_square / (2 * (1 - rho))
    return avg_waiting


with open(r'C:\Users\elira\PycharmProjects\redirected_calls\df_summary_result_lin_total_spread_15_10_a.pkl', 'rb') as f:
    df_summary = pkl.load(f)

for ind in range(df_summary.shape[0]):
    df_summary.loc[ind, 'exact_mg1'] = avg_waiting_time(df_summary.loc[ind, 'Arrival_0'], np.array(
        [df_summary.loc[ind, 'mu_0'], df_summary.loc[ind, 'mu_1']]))

df_summary['error_lin_exact'] = df_summary['exact_mg1'] - df_summary['avg_waiting_lin']
df_summary['error_lin_est'] = df_summary['avg_waiting_estimated'] - df_summary['avg_waiting_lin']

df_summary_09 = df_summary.loc[df_summary['Arrival_0'] == 0.92, :].reset_index()

mu_vals = np.arange(start=1.3, stop=3, step=0.3)

x = np.zeros((mu_vals.shape[0], mu_vals.shape[0]))
y = np.zeros((mu_vals.shape[0], mu_vals.shape[0]))
exact_val = np.zeros((mu_vals.shape[0], mu_vals.shape[0]))
estimated_val = np.zeros((mu_vals.shape[0], mu_vals.shape[0]))
for ind_mu0, mu0 in enumerate(mu_vals):
    for ind_mu1, mu1 in enumerate(mu_vals):
        x[ind_mu0, ind_mu1] = mu0
        y[ind_mu0, ind_mu1] = mu1
        exact_val[ind_mu0, ind_mu1] = df_summary_09.loc[
            (df_summary_09['mu_0'] == mu0) & (df_summary_09['mu_1'] == mu1), 'exact_mg1']
        estimated_val[ind_mu0, ind_mu1] = df_summary_09.loc[
            (df_summary_09['mu_0'] == mu0) & (df_summary_09['mu_1'] == mu1), 'avg_waiting_estimated']
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, exact_val, edgecolor='Black',
                label='real', alpha=1, rstride=1, cstride=1, linewidth=0.5, cmap='winter',
                antialiased=True)

ax.plot_surface(x, y, estimated_val, edgecolor='Blue',
                label='approx', alpha=1, rstride=1, cstride=1, linewidth=0.5, cmap='autumn',
                antialiased=True)

ax.set_title('Queue Length')
plt.xlabel('Mu0')
plt.ylabel('Mu1')

plt.show()