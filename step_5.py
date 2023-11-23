import numpy as np
import matplotlib.pyplot as plt
from step_1 import raw_data


def partial_corr(corr, x, y, q):
    return (corr[x, y] - corr[x, q] * corr[q, y]) / np.sqrt((1 - corr[x, q] ** 2) * (1 - corr[q, y] ** 2))


def correlation_analysis(corr, a, b, c):
    r_ab = partial_corr(corr, a, b, c)
    r_ac = partial_corr(corr, a, c, b)
    r_bc = partial_corr(corr, b, c, a)

    print(f"Часткова кореляція між {a} та {b} без урахування {c}: {r_ab}")
    print(f"Часткова кореляція між {a} та {c} без урахування {b}: {r_ac}")
    print(f"Часткова кореляція між {b} та {c} без урахування {a}: {r_bc}")

    r_abc = np.sqrt(
        (corr[a, b] ** 2 + corr[a, c] ** 2 - 2 * corr[a, b] * corr[a, c] * corr[b, c]) / (1 - corr[b, c] ** 2))

    print(f"множинний коефіцієнт кореляції для параметра a, при лінійному двофакторному зв’язку з параметрами b, c та: {r_abc}")

    r_abc = 1 - (1 - r_abc ** 2) * (1 - r_ab ** 2) * (1 - r_ac ** 2)

    print(f"множинний коефіцієнт кореляції для параметра a, при лінійному трифакторному зв’язку з параметрами b, c та: {r_abc}")


if __name__ == "__main__":
    norm_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    correlation_matrix = np.corrcoef(norm_data.T)
    np.fill_diagonal(correlation_matrix, 0)
    indices = np.unravel_index(np.argsort(correlation_matrix, axis=None)[-3:], correlation_matrix.shape)

    most_correlated_lines = [(indices[0][i], indices[1][i]) for i in range(3)]
    most_correlated_lines_idx = set([i for p in most_correlated_lines for i in p])

    correlation_analysis(correlation_matrix, *most_correlated_lines_idx)
