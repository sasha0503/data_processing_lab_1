import numpy as np
from scipy.stats import f_oneway
from step_1 import raw_data


def analyze_single_factor_anova(data, num_items=12):
    # Single-factor analysis. The null hypothesis is that the means of the samples are equal.
    return f_oneway(*[data[:, i] for i in range(num_items)])


def analyze_two_factors(data):
    reshaped_data = data.reshape(5, 12, 1000)
    average = np.mean(reshaped_data, axis=2)

    x_sum = np.sum(average, axis=1)
    y_sum = np.sum(average, axis=0)
    Q1 = np.sum(average ** 2)
    Q2 = np.sum(x_sum ** 2) / 5
    Q3 = np.sum(y_sum ** 2) / 12
    Q4 = (np.sum(x_sum) ** 2) / (5 * 12)

    S0 = (Q1 - Q2 - Q3 + Q4) / (5 -1) * (12 - 1)
    Sa = (Q2 - Q4) / (5 - 1)
    Sb = (Q3 - Q4) / (12 - 1)

    print(f"S0: {round(S0, 2)}")
    print(f"Sa: {round(Sa, 2)}")
    print(f"Sb: {round(Sb, 2)}")


if __name__ == "__main__":
    print("#################")
    print("Performing Single-factor Analysis of Variance")
    stat, p_value = analyze_single_factor_anova(raw_data)
    print('Statistics: %.3f, p-value: %.3f' % (stat, p_value))

    print("#################"
          "\nPerforming Two-factor Analysis of Variance")
    analyze_two_factors(raw_data)
