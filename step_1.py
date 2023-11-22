import base64

from io import BytesIO
from statistics import mode

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import normaltest
from tqdm import tqdm

data_file = "A30.txt"
raw_data = np.loadtxt(data_file, delimiter=',')


def get_stat(data):
    mean = np.mean(data)
    harmonic_mean = np.mean(1 / data)
    geometric_mean = np.exp(np.mean(np.log(np.abs(data))))
    dispersion = np.std(data)
    gini_diff = np.mean(np.abs(np.subtract.outer(data, data)))
    median = np.median(data)
    res_mode = mode(data)
    skewness = np.mean((data - mean) ** 3) / (dispersion ** 3)
    excess = np.mean((data - mean) ** 4) / (dispersion ** 4)

    standardized_data = (data - np.mean(data)) / np.std(data)
    stat, p = normaltest(standardized_data)

    # image in base64 histogram
    plt.hist(data, bins=100)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    hist_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    plt.plot(range(0, len(standardized_data) * 2, 2), standardized_data, color='red')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    cardiogram_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    data_dict = {
        "mean": mean,
        "harmonic_mean": harmonic_mean,
        "geometric_mean": geometric_mean,
        "dispersion": dispersion,
        "gini_diff": gini_diff,
        "median": median,
        "mode": res_mode,
        "skewness": skewness,
        "excess": excess,
        "histogram": hist_base64,
        "normaltest": (stat, p),
        "cardiogram": cardiogram_base64,
    }

    return data_dict


if __name__ == "__main__":
    with open("templates/html_template.html") as f:
        html_template = f.read().replace("\n", "")
    with open("templates/dataset_info_template.html") as f:
        dataset_info_template = f.read()

    dataset_html = ""
    raw_data = raw_data.T
    with tqdm(total=len(raw_data)) as pbar:
        for i, channel in enumerate(raw_data):
            channel_info = get_stat(channel)
            channel_info["channel_name"] = f"Channel {i + 1}"

            dataset_html += dataset_info_template.format(**channel_info)
            pbar.update(1)

    html = html_template.format(content=dataset_html)

    with open("step_1_result.html", "w") as f:
        f.write(html)
