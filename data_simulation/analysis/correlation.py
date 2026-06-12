from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Used for labelling axes and titles - gives mathematical notation equivalent to variable
mathematical_notation = {"Pivot_Energy": "$E_0$", "LP_Flux_Density": "$F_0$", "LP_Index": "$\\alpha$",
                         "LP_beta": "$\\beta$", "PLEC_Flux_Density": "$F_0$", "PLEC_IndexS": "$\Gamma$",
                         "PLEC_Exp_Index": "$b$", "PLEC_ExpfactorS": "$a$", "GLAT": "Latitude"}


# def correlation_matrices():




def analysis_correlation(sources, agn_params=False):

    # Find all possible combinations of parameters
    columns = list(sources.columns)
    variable_combinations = list(combinations(columns, 2))

    plt.rcParams["figure.figsize"] = (10, 14)

    if agn_params is True:
        num_plot_cols = 2
    else:
        num_plot_cols = 3

    # Find number of rows of subplots that will be in figure
    num_plot_rows = len(variable_combinations) // num_plot_cols

    fig, ax = plt.subplots(num_plot_rows, num_plot_cols)

    # Used to index subplots
    plot_indices = list(product(range(0, num_plot_rows), range(0, num_plot_cols)))

    # PLOT DATA AGAINST DATA

    # Plot data for each subplot
    for sp in range(len(plot_indices)):

        row, col = plot_indices[sp][0], plot_indices[sp][1]
        var1, var2 = variable_combinations[sp][0], variable_combinations[sp][1]

        # Check Kendall coefficient, as Pearson only determines if linear relationship.

        correlation_coefficient = str(round(sources[var1].corr(sources[var2]), 3))
        kendall_coefficient = str(round(sources[var1].corr(sources[var2], method='kendall'), 3))

        # Plot data
        ax[row, col].scatter(sources[var2], sources[var1], color='red', label="Pearson: " + correlation_coefficient +
                                                                              "\nKendall: " + kendall_coefficient, marker='+', s=8)

        # Subplot formatting
        ax[row, col].set_title(mathematical_notation[var1] + ' against ' + mathematical_notation[var2], fontsize=12)
        ax[row, col].set_xlabel(mathematical_notation[var2])
        ax[row, col].set_ylabel(mathematical_notation[var1])

        # Label with correlation coefficient
        # sources[[var1, var2]].corr(numeric_only=True)

        ax[row, col].legend(fontsize=8, loc='upper left')

    # Figure formatting

    fig.suptitle("Plotting AGN Parameters Against Each Other", fontsize=18, y=0.98)
    fig.tight_layout()

    fig.show()

    # LOG-LOG PLOTS

    fig, ax = plt.subplots(num_plot_rows, num_plot_cols)

    # Plot data for each subplot
    for sp in range(len(plot_indices)):

        row, col = plot_indices[sp][0], plot_indices[sp][1]
        var1, var2 = variable_combinations[sp][0], variable_combinations[sp][1]

        log_var1 = np.log(sources[var1])
        log_var2 = np.log(sources[var2])

        # Check Kendall coefficient, as Pearson only determines if linear relationship.

        correlation_coefficient = str(round(sources[var1].corr(sources[var2]), 3))
        kendall_coefficient = str(round(sources[var1].corr(sources[var2], method='kendall'), 3))

        # Plot data
        ax[row, col].scatter(log_var2, log_var1, color='red', label="Pearson: " + correlation_coefficient +
                                                                              "\nKendall: " + kendall_coefficient, marker='+', s=8)

        # Subplot formatting
        ax[row, col].set_title(mathematical_notation[var1] + ' against ' + mathematical_notation[var2], fontsize=12)
        ax[row, col].set_xlabel(mathematical_notation[var2])
        ax[row, col].set_ylabel(mathematical_notation[var1])

        ax[row, col].legend(fontsize=8, loc='upper left')

    # Figure formatting

    fig.suptitle("Log-Log Plotting AGN Parameters Against Each Other", fontsize=18, y=0.98)
    fig.tight_layout()

    fig.show()

    # CORRELATION MATRICES

    # Pearson

    corr = sources.corr()

    plt.figure(figsize=(13, 11))

    plt.title('Pearson Correlation Coefficient Matrix', fontsize=20)

    matrix_labels = [mathematical_notation[k] for k in corr.columns.values]

    sns.heatmap(np.abs(corr), xticklabels=matrix_labels, yticklabels=matrix_labels, annot=corr, cmap='Greens')

    plt.show()

    # Kendall Rank

    corr = sources.corr(method='kendall')

    plt.figure(figsize=(13, 11))

    plt.title('Kendall Rank Correlation Coefficient Matrix', fontsize=20)

    matrix_labels = [mathematical_notation[k] for k in corr.columns.values]

    # N.B. This is very important - took the absolute value to highlight suggestions of strong correlation, but
    # continued to label with + and - indicating positive or negative correlation
    sns.heatmap(np.abs(corr), xticklabels=matrix_labels, yticklabels=matrix_labels, annot=corr, cmap='Blues')

    plt.show()
