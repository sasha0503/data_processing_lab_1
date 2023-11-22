# Correlation analysis
# Carry out the main stages of correlation analysis.

import numpy as np
import matplotlib.pyplot as plt
from step_1 import raw_data

corr = np.corrcoef(raw_data.T)


def correlation_analysis(corr, a, b, c, d):
    print("\nCorrelation analysis")
    r_ab = (corr[a, b] - corr[a, c] * corr[b, c]) / np.sqrt((1 - corr[a, c] ** 2) * (1 - corr[b, c] ** 2))
    print(f"Partial correlation coefficient between attributes {a} and {b} without "
          f"taking into account the influence of {c}: {r_ab}")

    r_ac = (corr[a, c] - corr[a, b] * corr[b, c]) / np.sqrt((1 - corr[a, b] ** 2) * (1 - corr[b, c] ** 2))
    print(f"Partial correlation coefficient between attributes {a} and {c} without "
            f"taking into account the influence of {b}: {r_ac}")

    # Find the partial correlation coefficient between attributes a and b without taking into account the influence of c and d:
    r_abcd = (r_ab - r_ac * corr[b, d]) / np.sqrt((1 - r_ac ** 2) * (1 - corr[b, d] ** 2))
    print(f"Partial correlation coefficient between attributes {a} and {b} without "
            f"taking into account the influence of {c} and {d}: {r_abcd}")

    # Find the partial correlation coefficient between attributes a and c without taking into account the influence of b and d:
    r_acbd = (r_ac - r_ab * corr[b, d]) / np.sqrt((1 - r_ab ** 2) * (1 - corr[b, d] ** 2))
    print(f"Partial correlation coefficient between attributes {a} and {c} without "
            f"taking into account the influence of {b} and {d}: {r_acbd}")

    # Find the partial correlation coefficient between attributes a and d without taking into account the influence of b and c:
    r_adbc = (corr[a, d] - corr[a, b] * corr[b, d]) / np.sqrt((1 - corr[a, b] ** 2) * (1 - corr[b, d] ** 2))
    print(f"Partial correlation coefficient between attributes {a} and {d} without "
            f"taking into account the influence of {b} and {c}: {r_adbc}")

    # Find the multiple correlation coefficient for the parameter a, given linear two-factor relationship with
    # parameters b and c
    r_abc = np.sqrt((corr[a, b] ** 2 + corr[a, c] ** 2 - 2 * corr[a, b] * corr[a, c] * corr[b, c]) / (1 - corr[b, c] ** 2))
    print(f"Multiple correlation coefficient for the parameter {a}, given "
            f"linear two-factor relationship with parameters {b} and {c}: {r_abc}")

    # Find the multiple correlation coefficient for the parameter a, given
    # linear three-factor relationship with parameters b, c and d
    r_abcd = 1 - (1 - r_abc ** 2) * (1 - corr[b, d] ** 2) * (1 - r_ab ** 2) * (1 - r_ac ** 2)
    print(f"Multiple correlation coefficient for the parameter {a}, given "
            f"linear three-factor relationship with parameters {b}, {c} and {d}: {r_abcd}")


def normalize(data):
    # Normalize the data
    return (data - np.mean(data)) / np.std(data)


def plot_corr_matrix(data):
    # Plot the correlation matrix using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(corr, cmap='rainbow', interpolation='nearest')
    # add number in each cell
    for x in range(corr.shape[0]):
        for y in range(corr.shape[1]):
            plt.text(x, y, '%.2f' % corr[x, y], horizontalalignment='center', verticalalignment='center')
    plt.colorbar()
    plt.show()
    return corr


if __name__ == "__main__":
    norm_data = normalize(raw_data)
    plot_corr_matrix(norm_data)
    correlation_analysis(corr, a=1, b=3, c=5, d=11)



