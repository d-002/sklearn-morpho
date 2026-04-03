import csv
import numpy as np

class WdbcRow:
    def __init__(self, row: list[str]) -> None:
        self.id = int(row[0])
        self.diagnosis = row[1] == 'M'
        self.characteristics = np.array(row[2:], dtype=np.float32)

def dataset_wdbc(path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse and format data from a CSV file from the Wisconsin Breast Cancer
    Dataset (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

    param path:     Path to the file to open
    param **kwargs: Special arguments to give to the csv reader

    return:         A tuple [X, Y], where X is the set of data points and Y is
                    their associated labels
    """

    rows: list[WdbcRow] = []

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, **kwargs)
        for row in reader:
            rows.append(WdbcRow(row))

    return np.array([row.characteristics for row in rows]), \
            np.array([row.diagnosis for row in rows], dtype=np.bool)
