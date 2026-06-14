import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error


def fitting_agn_pivot_energy_spectral_slope_relation(var1, var2):

    plt.rcParams["figure.figsize"] = (10, 5)

    log_var1, log_var2 = np.log(var1), np.log(var2)

    # Create plot

    fig, ax = plt.subplots(1, 2)

    # Plot data

    ax[0].scatter(var1, var2, s=4, marker='+')
    ax[1].scatter(log_var1, log_var2, s=4, marker="+")

    # Fitting

    x_values = np.linspace(np.min(var1), np.max(var1), 1000)
    log_x_values = np.log(x_values)

    # Linear Fit
    # m, c = np.polyfit(log_var1, log_var2, deg=1)
    #
    # ax[0].plot(x_values, (x_values ** m) * np.exp(c), label='Log Linear', color='red')
    #
    # ax[1].plot(log_x_values, m * log_x_values + c, color='red', label='Linear')
    #
    # linear_residuals = log_var2 - (m * log_var1 + c)
    #
    # print("RMSE of Linear Fit: {}".format(root_mean_squared_error(log_var2, m * log_var1 + c)))
    #
    # print('m: {} c: {}'.format(m, c))
    #
    # # Quadratic fit
    #
    # quadratic = np.polynomial.Polynomial.fit(log_var1, log_var2, deg=2)
    #
    # ax[0].plot(x_values, np.exp(quadratic(log_x_values)), color='green',
    #            label='Log Quadratic', linestyle='-.')
    # ax[1].plot(log_x_values, quadratic(log_x_values), color='green',
    #            label='Quadratic', linestyle='-.')
    #
    # quadratic_residuals = log_var2 - quadratic(log_var1)
    #
    # print("RMSE of Quadratic Fit: {}".format(root_mean_squared_error(log_var2, quadratic(log_var1))))
    # print('a: {} b: {} c: {}'.format(quadratic.coef[2], quadratic.coef[1], quadratic.coef[0]))

    labels = ["Linear", "Quadratic", "Cubic", "Quartic"]
    colours = ["red", "green", "orange", "purple"]

    rs = []

    for degree in range(1, 5):

        idx = degree - 1

        polynomial = np.polynomial.Polynomial.fit(log_var1, log_var2, deg=degree)

        ax[0].plot(x_values, np.exp(polynomial(log_x_values)), color=colours[idx], label="Log {}".format(labels[idx]),
                   linestyle='-.')
        ax[1].plot(log_x_values, polynomial(log_x_values), color=colours[idx], label=labels[idx], linestyle="-.")

        rs.append(log_var2 - polynomial(log_var2))

        print(["{} : {}".format(chr((degree - x) + 97), polynomial.coef[x]) for x in range(degree, -1, -1)])
        print("RMSE of {}: {}".format(labels[idx], root_mean_squared_error(log_var2, polynomial(log_var1))))

    # # Cubic fit
    # a, b, c, d = np.polyfit(log_var1, log_var2, deg=3)
    #
    #
    # ax[0].plot(exp_x_values, np.e ** ((a * log_x_values ** 3) + (b * log_x_values ** 2) + (c * log_x_values) + d),
    #            linestyle='--', label='Log Cubic', color='orange')
    #
    # ax[1].plot(log_x_values, (a * log_x_values ** 3) + (b * log_x_values ** 2) + (c * log_x_values) + d, color='orange',
    #            label='Cubic', linestyle='--')
    #
    # cubic_residuals = log_var2 - ((a * log_var1 ** 3) + (b * log_var1 ** 2) + (c * log_var1) + d)
    #
    # print("RMSE of Cubic Fit: {}".format(root_mean_squared_error(log_var2, ((a * log_var1 ** 3)
    #                                                                           + (b * log_var1 ** 2)
    #                                                                           + (c * log_var1) + d))))
    #
    #
    # print('a: {} b: {} c: {} d: {}'.format(a, b, c, d))
    #
    # # Quartic fit
    # a, b, c, d, e = np.polyfit(log_var1, log_var2, deg=4)
    #
    #
    # ax[0].plot(exp_x_values, np.e ** ((a * log_x_values ** 4) + (b * log_x_values ** 3) + (c * log_x_values ** 2) +
    #                                   (d * log_x_values) + e), color='purple', label='Log Quartic', linestyle=':')
    # ax[1].plot(log_x_values, (a * log_x_values ** 4) + (b * log_x_values ** 3) + (c * log_x_values ** 2) +
    #            (d * log_x_values) + e, color='purple', label='Quartic', linestyle=':')
    #
    # quartic_residuals = log_var2 - ((a * log_var1 ** 4) + (b * log_var1 ** 3) +
    #                                   (c * log_var1 ** 2) + (d * log_var1) + e)
    #
    # print("RMSE of Quartic Fit: {}".format(root_mean_squared_error(log_var2, ((a * log_var1 ** 4) +
    #                                                                             (b * log_var1 ** 3) +
    #                                                                             (c * log_var1 ** 2) +
    #                                                                             (d * log_var1) + e))))
    #
    # print('a: {} b: {} c: {} d: {} e: {}'.format(a, b, c, d, e))

    # Suggested fit
    # This paper states that the spectral index depends linearly on ln E - https://journals-aps-org.ezphost.dur.ac.uk/
    # prd/abstract/10.1103/k5dp-5str
    m, c = np.polyfit(log_var1, var2, deg=1)

    ax[0].plot(x_values, (m * log_x_values) + c, color='black', label='Suggested')
    ax[1].plot(log_x_values, np.log((log_x_values * m) + c), color='black', label='Suggested')

    # suggested_residuals = log_alphas - (np.log((log_pivot_energies * m) + c))
    suggested_residuals = log_var2 - (np.log((log_var1 * m) + c))

    print("RMSE of Suggested Fit: {}".format(root_mean_squared_error(log_var2, np.log((log_var1 * m) + c))))

    print('m: {} c: {}'.format(m, c))

    # Formatting

    ax[0].set_xlabel("$E_0$")
    ax[0].set_ylabel("$\\alpha$")
    ax[0].set_title("Pivot Energy vs Spectral Slope")
    ax[0].legend()

    ax[1].set_xlabel("log $E_0$")
    ax[1].set_ylabel("log $\\alpha$")
    ax[1].set_title("Log-Log Plot of Pivot Energy vs Spectral Slope")
    ax[1].legend()

    fig.suptitle("Fitting AGN $E_0$-$\\alpha$ Dependency")

    fig.tight_layout()

    fig.show()

    plt.close()

    # DO NOT DELETE BELOW
    #
    # # Plot residuals
    #
    # residuals = [linear_residuals, quadratic_residuals, cubic_residuals, quartic_residuals, suggested_residuals]
    #
    # plt.rcParams["figure.figsize"] = (25, 5)
    #
    # fig2, ax2 = plt.subplots(1, 5)
    #
    # for a in range(len(residuals)):
    #
    #     residuals_std = np.std(residuals[a], ddof=1)
    #
    #     normalised_residuals = residuals[a] / residuals_std
    #
    #     ax2[a].scatter(log_var1, normalised_residuals, s=4, label="$\sigma =$ {0:.3f}".format(residuals_std))
    #
    # # FORMATTING
    #
    # fig2.suptitle('Residuals')
    #
    # ax2[0].set_title('Linear Fit to Log-Log Plot')
    # ax2[1].set_title('Quadratic Fit to Log-Log Plot')
    # ax2[2].set_title('Cubic Fit to Log-Log Plot')
    # ax2[3].set_title('Quartic Fit to Log-Log Plot')
    # ax2[4].set_title('Suggested Fit to Log-Log Plot')
    #
    # for a in ax2:
    #
    #     a.set_xlabel("log $E_0$")
    #     a.set_ylabel("$y_i - \hat{y}_i$")
    #     a.legend()
    #
    # fig.tight_layout()
    #
    # fig2.show()
    #
    # plt.close()

# REFERENCES

# Numpy Documentation - https://numpy.org/doc/stable/user/index.html
# Scipy Documentation - https://docs.scipy.org/doc/scipy/tutorial/index.html
# Seaborn Heatmaps - https://stackoverflow.com/questions/50947776/plot-two-seaborn-heatmap-graphs-side-by-side
