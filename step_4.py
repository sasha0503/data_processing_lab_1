import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from step_1 import raw_data


def fourier_transform():
    # Perform the Fourier transform using the formulas from the lecture.
    # Calculate the amplitude spectrum and the phase spectrum.

    plt.figure(figsize=(20, 15))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        fft_data = fft(raw_data[:, i])
        frequency = fftfreq(len(raw_data[:, i]), 1 / 1000)[1]
        print(f"Frequencies: {frequency}")
        plt.plot(fft_data[:1000], color='red')
        plt.title(f"Channel {i + 1}")
    plt.show()


def inverse_fourier_transform():
    # Perform the inverse Fourier transform using the formulas from the lecture.
    # Compare the results with the original data.

    plt.figure(figsize=(20, 15))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        fft_data = fft(raw_data[:, i])
        inverse_data = np.fft.ifft(fft_data)
        plt.plot(inverse_data[:1000], color='red')
        plt.title(f"Channel {i + 1}")
    plt.show()


if __name__ == "__main__":
    fourier_transform()
    inverse_fourier_transform()