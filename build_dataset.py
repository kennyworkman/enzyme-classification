import os
from shutil import copyfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRE_DATA = "Xs_generator/Xs" #insert directory of preprocessed np files
LABEL_CSV = "Xs_generator/batch_classification.csv" #insert csv with label info

labeled_data = pd.read_csv(LABEL_CSV, sep='\t')

#split Xs files by 80/10/10 distribution into train, validate, test directories
#retrieve labels for each sample (make dictionary with path and label)
#create seperate numpy arrays for each

def retrieve_filename(directory):
    """
    Return list of filenames in given directory
    """
    return [filename for filename in os.listdir(directory)]

def split_files(filenames):
    """
    Split given list of filenames into train, val, test sets.
    """
    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    train, val, test = filenames[:split_1], filenames[split_1:split_2], filenames[split_2:]
    return train, val, test

def make_change(name):
    """
    Change into the directory NAME, making it if it does not exist
    """
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

def make(name):
    """
    Create directory NAME, only if it does not exist.
    """
    if not os.path.exists(name):
        os.mkdir(name)

def build_data_dir():
    """
    Add a data directory to project.
    """
    make_change("./data")
    make("train")
    make("validate")
    make("test")
    os.chdir("..")

def fill_dir(filenames, dir):
    """
    Fill given directory with given list of filenames.
    """
    for file in filenames:
        copyfile(os.path.join(PRE_DATA, file), os.path.join(dir, file))

def build_populate_data():
    """
    Builds new data directory in . that has train, val, test subdirs
    Define PRE_DATA path above
    """
    build_data_dir()
    train_filenames, val_filenames, test_filenames = split_files(retrieve_filename(PRE_DATA))
    fill_dir(train_filenames, "./data/train")
    fill_dir(val_filenames, "./data/validate")
    fill_dir(test_filenames, "./data/test")

def retrieve_labeled_data(dir):
    """
    Retrieves numpy array of samples and numpy array of labels within given dir
    Make sure LABEL_CSV is defined above.
    """
    files = retrieve_filename(dir)
    samples, labels = [], []
    for file in files:
        enzyme = file[:4]
        csv_row = labeled_data.loc[labeled_data['label'] == enzyme]
        label = csv_row.type.iloc[0]
        samples.append(np.load(os.path.join(dir, file)))
        labels.append(label)
    return np.array(samples), np.array(labels)

build_populate_data()
train_Xs, train_labels = retrieve_labeled_data("./data/train")
validate_Xs, train_Xs = retrieve_labeled_data("./data/validate")
test_Xs, train_Xs = retrieve_labeled_data("./data/test")

# def retrieve_paths(directory):
#     """
#     Return list of paths for each X (numpy data file)
#     """
#     return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.npy')]
#
# def retrieve_paths_dict(Xs):
#     """
#     Need to pass a list of PATHS for each data file
#     Returns Dictionary of PATH matched with label
#     """
#     data = {}
#     for path in Xs:
#         filename = os.path.basename(path)
#         name = filename[:4]
#         row = labeled_data.loc[labeled_data['label'] == name]
#         data[path] = row.type.iloc[0]
#     return data
#
# def retrieve_labeled_data(dict):
#     """
#     Pass a dict of paths and labels
#     Returns samples (#, x, y, z, channel) and labels (#,)
#     """
#     samples, labels = [], []
#     for path in dict:
#         samples.append(np.load(path))
#         labels.append(dict[path])
#     return np.array(samples), np.array(labels)


# paths = retrieve_paths('/Users/kenny/desktop/chem195/enzyme_classifier/Xs_generator/Xs')
# paths_dict = retrieve_paths_dict(paths)
#
# Xs, labels = retrieve_labeled_data(paths_dict)


#filename[:4]
