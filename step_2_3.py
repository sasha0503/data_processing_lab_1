from scipy.stats import f_oneway
from step_1 import raw_data


def analyze_single_factor_anova(data, num_items=12):
    # Single-factor analysis. The null hypothesis is that the means of the samples are equal.
    return f_oneway(*[data[:, i] for i in range(num_items)])


def analyze_two_factors(data):
    print("\nPerforming Two-factor Analysis of Variance")
    reshaped_data = data.reshape(5, 12, 1000)

    # Analyze two-factor ANOVA for each channel with each channel
    # Calculate the average value for each channel

    print("\nFactor A: Channels")
    for i in range(12):
        print(f"Channel {i + 1}")
        stat, p_value = analyze_single_factor_anova(reshaped_data[:, i, :], num_items=5)
        print('Statistics: %.3f, p-value: %.3f' % (stat, p_value))

    print("\nFactor B: Channels")
    for i in range(5):
        print(f"Channel {i + 1}")
        stat, p_value = analyze_single_factor_anova(reshaped_data[i, :, :], num_items=12)
        print('Statistics: %.3f, p-value: %.3f' % (stat, p_value))


if __name__ == "__main__":
    print("Performing Single-factor Analysis of Variance")
    stat, p_value = analyze_single_factor_anova(raw_data)
    print('Statistics: %.3f, p-value: %.3f' % (stat, p_value))

    # Perform Two-factor Analysis of Variance
    analyze_two_factors(raw_data)
