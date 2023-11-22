import numpy as np
import matplotlib.pyplot as plt
from step_4 import corr
from step_1 import raw_data


# Finding the eigenvalues of a correlation matrix
def calc_eigenvalues(corr):
    print("\nFinding the eigenvalues of a correlation matrix")
    # Calculate the eigenvalues of the correlation matrix
    eig_vals, eig_vecs = np.linalg.eig(corr)
    # Sort the eigenvalues in descending order
    eig_vals = np.sort(eig_vals)[::-1]
    # Calculate the percentage of the variance explained by each eigenvalue
    eig_vals = eig_vals / np.sum(eig_vals)
    # print table of eigenvalues, part eigenvalues and cumulative sum of eigenvalues for each 12 channels
    print("Eigenvalues\tPart eigenvalues\tCumulative sum")
    for i in range(12):
        print(f"{eig_vals[i]:.4f}\t\t\t{eig_vals[i] / np.sum(eig_vals):.4f}\t\t\t{np.sum(eig_vals[:i + 1]):.4f}")

    # Plot the eigenvalues
    plt.figure(figsize=(10, 10))
    plt.plot(eig_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()
    return eig_vals


if __name__ == "__main__":
    eig_val = calc_eigenvalues(corr)
    # Find the number of principal components that explain 90% of the variance
    print(f"\nNumber of principal components that explain 90% of the variance: "
            f"{np.where(np.cumsum(eig_val) >= 0.9)[0][0] + 1}")

    # Calculating eigenvectors
    print("\nCalculating eigenvectors")
    eig_vals, eig_vecs = np.linalg.eig(corr)

    # Building basic component gifs for 2 first comp, checking properties
    print("\nBuilding basic component gifs, checking properties")
    # Calculate the first three basic components
    basic_components = np.dot(raw_data, eig_vecs[:, :3])
    # Plot the basic components[0]
    plt.figure(figsize=(25, 5))
    plt.plot(basic_components[:, 0], color='red')
    plt.show()
