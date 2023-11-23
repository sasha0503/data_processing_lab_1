import numpy as np
import matplotlib.pyplot as plt
from step_1 import raw_data

if __name__ == "__main__":
    norm_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    correlation_matrix = np.corrcoef(norm_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    relative_eigenvalues = np.sort(eigenvalues)[::-1]
    relative_eigenvalues = relative_eigenvalues / np.sum(relative_eigenvalues)

    plt.figure(figsize=(10, 10))
    plt.plot(relative_eigenvalues, 'ro-', linewidth=2)
    plt.ylabel('Eigenvalue')

    plt.show()
    plt.close()

    max_prod = 0
    for vector_1 in eigenvectors.T:
        for vector_2 in eigenvectors.T:
            if not np.array_equal(vector_1, vector_2):
                max_prod = max(max_prod, np.dot(vector_1, vector_2))

    print(f"Max product of eigenvectors: {max_prod}")

    for vector in eigenvectors.T:
        print(f"vector*vector: {np.dot(vector, vector)}")

    main_components = np.dot(norm_data, eigenvectors[:, :3])
    plt.figure(figsize=(18, 12))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(main_components[:, i])
        plt.title(f"Main component {i + 1}")
    plt.tight_layout()
    plt.show()
