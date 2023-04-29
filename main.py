import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

def interpolate_outliers_consecutive_diff_iqr(df, multiplier=1, verbose=False):
    if verbose:
        print(f'Using {multiplier}*IQR')
    y_columns = df.loc[:, df.columns.str.endswith('Y')]
    for column in y_columns:
        diff = np.abs(df[column] - df[column].shift())
        Q1 = diff.quantile(0.25)
        Q3 = diff.quantile(0.75)
        IQR = Q3 - Q1
        threshold = IQR * multiplier
        outliers_mask = (diff > (Q3 + threshold)) | (diff < (Q1 - threshold))
        df.loc[outliers_mask, column] = np.nan
        df[column] = df[column].interpolate()
        if verbose:
            print(f"Column: {column}")
            print(f"Percentage of outliers: {sum(outliers_mask) / len(df) * 100:.2f}%")
    return df

def smooth(df, window_size=500):
    y_columns = df.loc[:, df.columns.str.endswith('Y')]
    for column in y_columns:
        # Let x_i be the i-th element of the original data (i.e., df[column]), and let y_i be the i-th element of the smoothed data (i.e., the rolling mean of the original data). The rolling mean of window size W is calculated as follows:
        # y_i = (x_i + x_(i-1) + x_(i-2) + ... + x_(i-W+1)) / min(i, W)
        df[column] = df[column].rolling(
            window_size, min_periods=1
        ).mean()
    return df

def residuals_func(p, time, data):
    R_max, k_off = p
    return data - exponential_decay(time, R_max, k_off)

def calculate_residuals_and_chisq(observed, predicted):
    residuals = observed - predicted
    squared_residuals = residuals ** 2
    chisq = np.sum(squared_residuals)
    return residuals, chisq

def calculate_reduced_chisq(chisq, data_points, num_params):
    dof = data_points - num_params
    reduced_chisq = chisq / dof
    return reduced_chisq

def langmuir_binding(t, R_max, k_obs):
    return R_max * (1 - np.exp(-k_obs * t))

def langmuir_residuals(params, t, observed):
    R_max, k_obs = params
    predicted = langmuir_binding(t, R_max, k_obs)
    return observed - predicted

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def r_squared_func(observed, expected):
    residuals = observed - expected
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared_value = 1 - (ss_res / ss_tot)
    return r_squared_value

def hyperbolic_binding_equation(x, R_max, Kd):
    return R_max * x / (Kd + x)

def residuals_func_hyperbolic(params, concentration, R_max_values):
    R_max, Kd = params
    return R_max_values - hyperbolic_binding_equation(concentration, R_max, Kd)


# Load data
df = pd.read_csv('data.tsv', delimiter='\t')

# Create output directories if they don't exist
os.makedirs('output_data/', exist_ok=True)
os.makedirs('output_data/k_obs_vs_C/', exist_ok=True)
os.makedirs('output_data/langmuir_curve_fits/', exist_ok=True)
os.makedirs('output_data/dissociation_curve_fits/', exist_ok=True)
os.makedirs('output_data/R_max_vs_concentration/', exist_ok=True)

# Plot sensogram
plt.plot(df['Cycle: 14  IgD  500 nM_X'], df['Cycle: 14  IgD  500 nM_Y'], label='500nM', color='blue')
plt.plot(df['Cycle: 15  IgD  250 nM_X'], df['Cycle: 15  IgD  250 nM_Y'], label='250nM', color='red')
plt.plot(df['Cycle: 16  IgD  125 nM_X'], df['Cycle: 16  IgD  125 nM_Y'], label='125nM', color='green')
plt.plot(df['Cycle: 17  IgD  62 nM_X'], df['Cycle: 17  IgD  62 nM_Y'], label='62nM', color='yellow')
plt.plot(df['Cycle: 18  IgD  31 nM_X'], df['Cycle: 18  IgD  31 nM_Y'], label='31nM', color='magenta')
plt.plot(df['Cycle: 19  IgD  16 nM_X'], df['Cycle: 19  IgD  16 nM_Y'], label='16nM', color='cyan')
plt.plot(df['Cycle: 20  IgD  8 nM_X'], df['Cycle: 20  IgD  8 nM_Y'], label='8nM', color='black')
plt.xlabel('Time(s)')
plt.ylabel('RU')
plt.grid(True)
plot_x_lim = (min(df['Cycle: 14  IgD  500 nM_X']) -50, max(df['Cycle: 14  IgD  500 nM_X']) +50)
plot_y_lim = (-5, 45)
plt.xlim(*plot_x_lim)
plt.ylim(*plot_y_lim)
plt.legend()
plt.savefig('output_data/sensogram.png')
plt.close()

# Drop NaN
df.dropna(inplace=True)
# Drop 8nM and 16nM data
df.drop(
    columns = ['Cycle: 19  IgD  16 nM_Y', 'Cycle: 20  IgD  8 nM_Y'],
    inplace = True
)
# We only keep the first X axis since they're all the same
df.drop(
    columns = df.columns[df.columns.str.endswith('X')][1:],
    inplace = True
)
df.rename(
    columns={
        df.columns[df.columns.str.endswith('X')][0]: 'time'
    },
    inplace = True
)
# Interpolate over Y values where abs difference to previous is outlier
df = interpolate_outliers_consecutive_diff_iqr(df)
# Smoothing
df = smooth(df)

# Plot processed sensogram
plt.plot(df['time'], df['Cycle: 14  IgD  500 nM_Y'], label='500nM', color='blue')
plt.plot(df['time'], df['Cycle: 15  IgD  250 nM_Y'], label='250nM', color='red')
plt.plot(df['time'], df['Cycle: 16  IgD  125 nM_Y'], label='125nM', color='green')
plt.plot(df['time'], df['Cycle: 17  IgD  62 nM_Y'], label='62nM', color='yellow')
plt.plot(df['time'], df['Cycle: 18  IgD  31 nM_Y'], label='31nM', color='magenta')
plt.xlabel('Time(s)')
plt.ylabel('RU')
plt.grid(True)
plt.xlim(*plot_x_lim)
plt.ylim(*plot_y_lim)
plt.legend()
plt.savefig('output_data/processed_data_sensogram.png')
plt.close()

y_columns = df.loc[:, df.columns.str.endswith('Y')]

concentration_values = [500, 250, 125, 62, 31]

# Split association and dissociation data
Rmax_row_index = df['Cycle: 14  IgD  500 nM_Y'].idxmax()
df.drop(df[df['time'] < 15].index, inplace=True)
df_assoc = df.drop(df.index[df.index > Rmax_row_index])
df_dissoc = df.drop(df.index[df.index <= Rmax_row_index])

response_association = df_assoc.loc[:, df_assoc.columns.str.endswith('Y')]
time_association = df_assoc['time'].values

# Calculate k_obs values by fitting Langmuir binding model
k_obs_values = []
R_max_values = []
k_obs_stderr = []
chi_square_values = []

for i, column in enumerate(response_association.columns):
    R_max = max(response_association[column].values)
    initial_guess = [R_max, 1.13e-02]
    result = least_squares(langmuir_residuals, initial_guess, args=(time_association, response_association[column].values))
    R_max, k_obs = result.x
    R_max_values.append(R_max)
    k_obs_values.append(k_obs)
    jacobian = result.jac
    hessian = np.dot(jacobian.T, jacobian)
    cov_matrix = np.linalg.inv(hessian)
    stderr = np.sqrt(np.diag(cov_matrix))
    k_obs_stderr.append(stderr)

    # Calculate chi-square
    observed = response_association[column].values
    expected = langmuir_binding(time_association, R_max, k_obs)
    residuals = observed - expected
    chi_square = np.sum((residuals ** 2) / expected)
    chi_square_values.append(chi_square)
    reduced_chi_square = calculate_reduced_chisq(chi_square, len(observed), 2)
    r_squared = r_squared_func(observed, expected)

    # Export k_obs values with the standard deviation and chi-square
    with open(f'output_data/langmuir_curve_fits/k_obs_{concentration_values[i]}_nM.txt', 'w') as f:
        f.write('Langmuir binding model: R(t) = R_max * (1 - exp(-k_obs * t))\n')
        f.write('Units: R(t) (Response Units), R_max (Response Units), k_obs (1/s), t (s)\n\n')
        f.write(f'R_max: {R_max}, k_obs: {k_obs}\n')
        f.write(f'Standard deviation (sqrt of diagonal elements of the covariance matrix):\nRmax: {stderr[0]}, k_obs:{stderr[1]}\n')
        f.write(f'Chi-square: {chi_square} Reduced Chi-square: {reduced_chi_square}')
        f.write(f'R-squared: {r_squared}')

    # Plot individual Langmuir curve fit
    plt.figure()
    plt.plot(time_association, response_association[column].values, 'o', label='Data')
    plt.plot(time_association, langmuir_binding(time_association, R_max, k_obs), '-', label='Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Langmuir curve fit for {concentration_values[i]} nM')
    plt.legend()
    plt.grid(True)
    plt.text(0.1 * max(time_association), 0.8 * max(response_association[column].values), f'k_obs: {k_obs:.2e}', fontsize=9)
    plt.savefig(f'output_data/langmuir_curve_fits/{concentration_values[i]}_nM.png')

# Plot all the curves and data
plt.figure()
for i, column in enumerate(response_association.columns):
    R_max = R_max_values[i]
    plt.plot(time_association, response_association[column].values, 'o', label=f'{concentration_values[i]} nM')
    plt.plot(time_association, langmuir_binding(time_association, R_max, k_obs_values[i]), '-', label=f'Fit {concentration_values[i]} nM')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('All Langmuir curve fits')
plt.legend()
plt.grid(True)
plt.savefig('output_data/langmuir_curve_fits/all.png')


# Calculate k_on and k_off using linear regression in log space
slope, intercept, r_value, p_value, std_err = linregress(np.log(concentration_values), np.log(k_obs_values))
k_on_log = np.exp(intercept)
k_off_log = slope
# Calculate residuals and chi-squared for log space regression
k_obs_predicted_log = k_on_log * np.array(concentration_values) ** k_off_log
residuals_log, chisq_log = calculate_residuals_and_chisq(k_obs_values, k_obs_predicted_log)
r_squared_log = r_squared_func(np.array(k_obs_values), k_obs_predicted_log)
reduced_chisq_log = calculate_reduced_chisq(chisq_log, len(k_obs_values), 2)
# Plot the data and curve
plt.figure()
plt.plot(np.log(concentration_values), np.log(k_obs_values), 'o', label='Data')
x_values = np.linspace(0, max(np.log(concentration_values)) + 1, 100)
plt.plot(x_values, intercept + slope * x_values, '-', label='Fit')
plt.xlabel('log(Concentration) (log(nM))')
plt.ylabel('log(k_obs) (log(1/s))')
plt.title('Linear Regression in Log Space')
plt.legend()
plt.grid(True)
plt.xlim(0, max(np.log(concentration_values)) + 1)
plt.ylim(intercept - 0.2, max(np.log(k_obs_values)) + 1)
plt.annotate(f"Intercept {round(k_on_log, 5)}", xy=(0.1, -5.55), xytext=(0.5, -5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9)
plt.text(min(np.log(concentration_values)), intercept + 1.5, f'k_on: {k_on_log:.2e} 1/s/nM')
plt.savefig('output_data/k_obs_vs_C/log_regression.png')
plt.close()

# Calculate k_on and k_off using linear regression in normal space
slope, intercept, r_value, p_value, std_err = linregress(concentration_values, k_obs_values)
k_on_normal = slope
k_off_normal = intercept
r_squared_normal = r_value ** 2
# Calculate residuals and chi-squared for normal space regression
k_obs_predicted_normal = k_on_normal * np.array(concentration_values) + k_off_normal
residuals_normal, chisq_normal = calculate_residuals_and_chisq(k_obs_values, k_obs_predicted_normal)
reduced_chisq_normal = calculate_reduced_chisq(chisq_normal, len(k_obs_values), 2)

# Plot the data and curve
plt.figure()
plt.plot(concentration_values, k_obs_values, 'o', label='Data')
x_values = np.linspace(0, max(concentration_values) + 10, 100)
plt.plot(x_values, intercept + slope * x_values, '-', label='Fit')
plt.xlabel('Concentration (nM)')
plt.ylabel('k_obs (1/s)')
plt.title('Linear Regression in Original Space')
plt.legend()
plt.grid(True)
plt.xlim(0, max(concentration_values) + 10)
plt.ylim(min(k_obs_values)-0.001, max(k_obs_values) + 0.001)
plt.annotate(f"Intercept {round(intercept, 5)}", xy=(15, intercept), xytext=(100, intercept),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9)
plt.annotate(f"Slope {round(slope, 5)}", xy=(max(concentration_values) * 0.5, intercept + slope * max(concentration_values) * 0.5),
             xytext=(230, 0.009),
             fontsize=9)
plt.savefig('output_data/k_obs_vs_C/normal_regression.png')
plt.close()

# Export k_on, k_off and all other relevant values, along with standard deviation where applicable
with open('output_data/k_obs_vs_C/normal_regression.txt', 'w') as f:
    f.write('Linear regression in original space: k_obs = k_on * C + k_off\n')
    f.write('Units: k_on (1/s/nM), k_off (1/s), Concentration (nM), k_obs (1/s)\n\n')
    f.write('Concentration (nM)\tk_obs (1/s)\tk_obs Standard Deviation\n')
    for i, value in enumerate(concentration_values):
        f.write(f'{value}\t{k_obs_values[i]}\t{k_obs_stderr[i]}\n')
    f.write(f'\nk_on: {k_on_normal}\n')
    f.write(f'k_off: {k_off_normal}\n')
    f.write('\nResiduals (observed - predicted):\n')
    for i, value in enumerate(concentration_values):
        f.write(f'{value}\t{residuals_normal[i]}\n')
    f.write(f'\nChi-squared: sum of squared residuals: {chisq_normal}\n')
    f.write(f'\nR-squared: {r_squared_normal}\n')
with open('output_data/k_obs_vs_C/log_regression.txt', 'w') as f:
    f.write('Linear regression in log space: ln(k_obs) = ln(k_on) + k_off * ln(C)\n')
    f.write('Units: k_on (1/s/nM), k_off (unitless), Concentration (nM), k_obs (1/s)\n\n')
    f.write('Concentration (nM)\tk_obs (1/s)\tk_obs Standard Deviation\n')
    for i, value in enumerate(concentration_values):
        f.write(f'{value}\t{k_obs_values[i]}\t{k_obs_stderr[i]}\n')
    f.write(f'\nk_on: {k_on_log}\n')
    f.write(f'k_off: {k_off_log}\n')
    f.write('\nResiduals (observed - predicted):\n')
    for i, value in enumerate(concentration_values):
        f.write(f'{value}\t{residuals_log[i]}\n')
    f.write(f'\nChi-squared: sum of squared residuals: {chisq_log}\n')
    f.write(f'\nR-squared: {r_squared_log}\n')

time_dissociation = df_dissoc['time'].values
response_dissociation = df_dissoc.loc[:, df_assoc.columns.str.endswith('Y')]

# Fit dissociation decay model and calculate k_off
predicted_k_off_values = []
k_off_stderr = []
R_max_values = []
chi_square_values = []

response_dissociation = df_dissoc.loc[:, df_dissoc.columns.str.endswith('Y')]
time_dissociation = df_dissoc['time'].values - min(df_dissoc['time'].values)

for i, column in enumerate(response_dissociation.columns):
    R_max = max(response_dissociation[column].values)
    initial_guess = [R_max, 0.0005]

    result = least_squares(residuals_func, initial_guess, args=(time_dissociation, response_dissociation[column].values), max_nfev=100000)
    R_max, k_off = result.x
    R_max_values.append(R_max)
    predicted_k_off_values.append(k_off)

    # Calculate stderr
    jacobian = result.jac
    hessian = np.dot(jacobian.T, jacobian)
    cov_matrix = np.linalg.inv(hessian)
    stderr = np.sqrt(np.diag(cov_matrix))
    k_off_stderr.append(stderr[1])

    # Calculate chi-square
    observed = response_dissociation[column].values
    expected = exponential_decay(time_dissociation, R_max, k_off)
    residuals = observed - expected
    chi_square = np.sum((residuals ** 2) / expected)
    chi_square_values.append(chi_square)
    reduced_chi_square = calculate_reduced_chisq(chi_square, len(observed), 2)
    r_squared = r_squared_func(observed, expected)

    # Save the values to a txt file for each column
    with open(f'output_data/dissociation_curve_fits/{concentration_values[i]}_nM.txt', 'w') as output_file:
        output_file.write(f'Concentration: {concentration_values[i]} nM\n')
        output_file.write(f'R_max: {R_max}\n')
        output_file.write(f'k_off: {k_off}\n')
        output_file.write(f'Standard error:\nR_max: {stderr[0]}, k_off: {stderr[1]}\n')
        output_file.write(f'Chi-square: {chi_square}\n')
        output_file.write(f'Reduced chi-square: {reduced_chi_square}\n')
        output_file.write(f'R-squared: {r_squared}\n')

    # Plot individual dissociation decay curve fit
    plt.figure()
    plt.plot(time_dissociation, response_dissociation[column].values, 'o', label='Data')
    plt.plot(time_dissociation, exponential_decay(time_dissociation, R_max, k_off), '-', label='Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Dissociation decay curve fit for {concentration_values[i]} nM')
    plt.legend()
    plt.grid(True)
    plt.text(0.1 * max(time_dissociation), 0.8 * max(response_dissociation[column].values), f'k_off: {k_off:.2e}', fontsize=9)
    plt.savefig(f'output_data/dissociation_curve_fits/{concentration_values[i]}_nM.png')

# Calculate mean k_off
mean_k_off = np.mean(predicted_k_off_values)
stderr_k_off = np.mean(k_off_stderr)

# Plot all the dissociation decay curves and data
plt.figure()
for i, column in enumerate(response_dissociation.columns):
    R_max = R_max_values[i]
    plt.plot(time_dissociation, response_dissociation[column].values, 'o', label=f'{concentration_values[i]} nM')
    plt.plot(time_dissociation, exponential_decay(time_dissociation, R_max, predicted_k_off_values[i]), '-', label=f'Fit {concentration_values[i]} nM')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('All dissociation decay curve fits')
plt.legend()
plt.grid(True)
plt.savefig('output_data/dissociation_curve_fits/all.png')

# Export predicted k_off values, mean k_off, and standard deviation
with open('output_data/dissociation_curve_fits/predicted_k_off.txt', 'w') as f:
    f.write('Dissociation decay model: R(t) = R_max * exp(-k_off * t)\n')
    f.write('Units: R(t) (Response Units), R_max (Response Units), k_off (1/s), t (s)\n')
    f.write('\n')
    f.write('Concentration (nM)\tPredicted k_off (1/s)\tStandard Error\tChi-square\n')
    for i, k_off in enumerate(predicted_k_off_values):
        f.write(f'{concentration_values[i]}\t{k_off:.6e}\t{k_off_stderr[i]:.6e}\t{chi_square_values[i]:.6e}\n')
    f.write('\n')
    f.write(f'Mean k_off: {mean_k_off:.6e} 1/s\n')
    f.write(f'Standard Error of k_off: {stderr_k_off:.6e} 1/s\n')

# Calculate and plot R_max vs Concentration
initial_guess = [max(R_max_values), 1]
result = least_squares(residuals_func_hyperbolic, initial_guess, args=(np.array(concentration_values), R_max_values))

R_max_opt, Kd_opt = result.x
jacobian = result.jac
hessian = np.dot(jacobian.T, jacobian)
cov_matrix = np.linalg.inv(hessian)
stderr = np.sqrt(np.diag(cov_matrix))

observed = R_max_values
expected = hyperbolic_binding_equation(np.array(concentration_values), R_max_opt, Kd_opt)
residuals = observed - expected
chi_square = np.sum((residuals ** 2) / expected)
reduced_chi_square = calculate_reduced_chisq(chi_square, len(R_max_values), 2)
r_squared = r_squared_func(observed, expected)

plt.figure()
plt.plot(concentration_values, R_max_values, 'o', markersize=8, linewidth=2)
plt.plot(concentration_values, expected, '-', label='Curve Fit')
plt.xlabel('Concentration (nM)')
plt.ylabel('R_max (Response Units)')
plt.title('R_max vs Concentration')
plt.legend()
plt.grid(True)
plt.savefig('output_data/R_max_vs_concentration/R_max_vs_concentration.png')
plt.close()

with open('output_data/R_max_vs_concentration/R_max_fit_results.txt', 'w') as f:
    f.write('Hyperbolic Binding Model: R = R_max * [C] / (Kd + [C])\n')
    f.write('Units: R (Response Units), R_max (Response Units), [C] (nM), Kd (nM)\n\n')
    f.write('Concentration (nM)\tR_max (Response Units)\tExpected R (Response Units)\n')
    for i, concentration in enumerate(concentration_values):
        f.write(f'{concentration}\t{R_max_values[i]}\t{expected[i]}\n')
    f.write('\n')
    f.write(f'R_max: {R_max_opt}\n')
    f.write(f'Kd: {Kd_opt}\n')
    f.write(f'Standard Error:\n R_max: {stderr[0]}, Kd: {stderr[1]}\n')
    f.write(f'Chi-square: {chi_square}\n')
    f.write(f'Reduced Chi-square: {reduced_chi_square}\n')
    f.write(f'R-squared: {r_squared}\n')
