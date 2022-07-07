import os
import csv
import numpy as np

def _make_dir(dir_name: str) -> str:
    """
    A utility function to create a directory of name <dir_name> if it does not already exist
    """
    if not os.path.isdir(dir_name):
        print('Creating new directory...')
        os.mkdir(dir_name)
    else:
        print('Directory already exists!')

    return dir_name


def _write_hist_csv(freq: list, bins: np.ndarray, filename: str) -> None:
    """
    A utility function to write out the histogram to a csv file
    """
    with open(filename + '.csv', 'w', newline='') as csvfile:
        histwriter = csv.writer(csvfile, delimiter=',')
        histwriter.writerow(('Value', 'Probability'))
        for i in range(np.size(freq)):
            histwriter.writerow((bins[i], freq[i]))
    print("Histogram .csv written")

    return

def _scale_image(image_data: np.ndarray, scale_to: type = np.uint8) -> np.ndarray:
    """
    A utility function to scale the data to a different datatype range
    Default converts to uint8
    Allows thumbnails and animated gif to show properly
    """
    assert image_data.dtype.type is not scale_to, f"Image data is already of type {scale_to.__name__}"

    if 'int' in scale_to.__name__:
        dtype_info = np.iinfo(scale_to)

    elif 'float' in scale_to.__name__:
        dtype_info = np.finfo(scale_to)

    else:
        raise ValueError("Please supply valid datatype to scale to")

    image_min = np.min(image_data)
    image_max = np.max(image_data)
    scale_grad = dtype_info.max / (image_max - image_min)
    scale_min = -scale_grad * image_min

    return np.floor(image_data * scale_grad + scale_min).astype(scale_to)

