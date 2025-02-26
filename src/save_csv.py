import csv

from src.utils import path_results, q_test, t_test, j_pred


def save_results(results: dict, file: str, path: str = path_results) -> None:
    """
    Save the results in a csv file

    :param results: dict of results
    :param file: file name
    :param path: path to save the file
    """
    with open(path + file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))


def load_results(results: dict, file: str, path: str = path_results) -> dict:
    """
    Load the results from a csv file

    :param results: dict of results
    :param file: file name
    :param path: path to save the file
    """
    with open(path + file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            results[row[0]] = row[1:]
    return results


def load_results_mul(file: str, path: str = path_results) -> dict:
    """
    Load the results from multiple csv files

    :param results: dict of results
    :param file: list of file names
    :param path: path to save the file
    """
    with open(path + file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        results = {key: [] for key in header}
        for row in reader:
            for i, key in enumerate(header):
                results[key].append(float(row[i]))
    return results
