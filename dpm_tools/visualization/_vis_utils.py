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


def _write_hist_csv(freq: list, bins: np.ndarray, filename: str):
    """
    A utility function to write out the histogram to a csv file
    """
    with open(filename + '.csv', 'w', newline='') as csvfile:
        histwriter = csv.writer(csvfile, delimiter=',')
        histwriter.writerow(('Value', 'Probability'))
        for i in range(np.size(freq)):
            histwriter.writerow((bins[i], freq[i]))
    print("Histogram .csv written")
