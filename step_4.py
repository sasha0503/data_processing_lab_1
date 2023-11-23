import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

from step_1 import raw_data


def frequency_analysis(data, analysis_type="fourier"):
    plt.figure(figsize=(18, 12))
    for i in range(data.shape[1]):
        plt.subplot(4, 3, i + 1)

        if analysis_type == "fourier":
            spectrum = fft(data[:, i])
            frequencies = fftfreq(len(data[:, i]), 1 / 1000)[1]
            print(f"Detected Frequencies: {frequencies}")

            plt.plot(spectrum[:1000], color='green')
            plt.title(f"Channel {i + 1} - Fourier Transform")

        elif analysis_type == "inverse_fourier":
            spectrum = fft(data[:, i])
            inverse_data = np.fft.ifft(spectrum)

            plt.plot(inverse_data[:1000], color='green')
            plt.plot(data[:, i][:1000], color='red')
            plt.title(f"Channel {i + 1} - Inverse Fourier Transform")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frequency_analysis(raw_data, analysis_type="fourier")
    frequency_analysis(raw_data, analysis_type="inverse_fourier")
