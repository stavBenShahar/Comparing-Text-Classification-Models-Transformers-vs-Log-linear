import os
import pickle
from functools import wraps
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


def get_data(categories=None, portion=1.) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    train_data_path = './pickles/data_train.pkl'
    test_data_path = './pickles/data_test.pkl'

    if not os.path.exists(train_data_path):
        data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                        random_state=21)
        with open(train_data_path, 'wb') as f:
            pickle.dump(data_train, f)
    else:
        with open(train_data_path, 'rb') as f:
            data_train = pickle.load(f)

    # Check if the testing data file exists
    if not os.path.exists(test_data_path):
        data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                       random_state=21)
        with open(test_data_path, 'wb') as f:
            pickle.dump(data_test, f)
    else:
        with open(test_data_path, 'rb') as f:
            data_test = pickle.load(f)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


def plot(accuracies: list[float], portions: list[float], model_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(portions, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Model: {model_name} -  Accuracy vs Portion of Data Used')
    plt.xlabel('Portion of Data')
    plt.ylabel('Accuracy')
    plt.xticks(portions, labels=[f'{int(p * 100)}%' for p in portions])
    plt.grid(True)
    save_dir = f'plots/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_accuracy_vs_portion.png')
    plt.savefig(save_path)
    plt.close()

def report_data(accuracies, losses, portions, model_name) -> None:
    if not (len(accuracies) == len(losses) == len(portions)):
        print("Error: The lengths of accuracies, losses, and portions lists must match.")
        return
    print(f'Model: {model_name} - Accuracy & Loss for every epoch in a given portion')
    for i, portion in enumerate(portions):
        print(f'\tPortion: {portion}')
        for j, (accuracy, loss) in enumerate(zip(accuracies[i], losses[i])):
            print(f'\t\tEpoch: {j}, Accuracy: {accuracy:.6f}, Loss: {loss:.6f}')


def announce(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        title = f'{func.__module__}.{func.__name__}'
        print(title.center(len(title) + 8, " ").center(50, "="))
        return func(*args, **kwargs)

    return wrapper


def multiproc(func):
    def wrapper(*args, **kwargs):
        multiprocessing.Process(target=func, args=args, kwargs=kwargs).start()

    return wrapper


def threadify(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()

    return wrapper


def split_list(lst: list, splits: int) -> list[list]:
    n = len(lst)
    if splits > n:
        splits = n
    delta = n // splits
    res = []
    for start, end in zip(range(0, n, delta), range(delta, n, delta)):
        res.append(lst[start:end])
    res.append(lst[start + delta:])
    return res


__all__ = [
    "get_data",
    "plot",
    "announce",
    "threadify",
    "multiproc",
    "split_list"
]
