import os
from shutil import copyfile, rmtree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRE_DATA = "Xs_generator/Xs" #insert directory of preprocessed np files
LABEL_CSV = "Xs_generator/batch_classification.csv" #insert csv with label info

labeled_data = pd.read_csv(LABEL_CSV, sep='\t')

"""
split Xs files by 80/10/10 distribution into train, validate, test directories
Option to then generate numpy array of data or dictionary of references (works better with keras generate)
"""

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
    make("validation")
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
    print("Building data directories and split subdirectories.")
    build_data_dir()
    train_filenames, val_filenames, test_filenames = split_files(retrieve_filename(PRE_DATA))
    print("Filling training directory...")
    fill_dir(train_filenames, "./data/train")
    print("Filling validation directory...")
    fill_dir(val_filenames, "./data/validation")
    print("Filling test directory...")
    fill_dir(test_filenames, "./data/test")

def retrieve_labeled_data(dir):
    """
    Retrieves numpy array of samples and numpy array of labels within given dir
    Make sure LABEL_CSV is defined above.
    """
    files = retrieve_filename(dir)
    samples, labels = [], []
    for enzyme in files:
        csv_row = labeled_data.loc[labeled_data['label'] == enzyme]
        label = csv_row.type.iloc[0]
        samples.append(np.load(os.path.join(dir, file)))
        labels.append(label)
    return np.array(samples), np.array(labels)

def remove_extension(list):
    """
    Remove file extension from a list of samples
    """
    return [file.split('.')[0] for file in list]

def retrieve_partioned_data(dir):
    """
    Retrieves dictionary of partioned sample IDs from given data directory
    Make sure PATH is passed as string
    """
    train = retrieve_filename(os.path.join(dir, 'train'))
    validation = retrieve_filename(os.path.join(dir, 'validation'))
    test = retrieve_filename(os.path.join(dir, 'test'))
    train, validation, test = remove_extension(train), remove_extension(validation), remove_extension(test)
    return {'train': train, 'validation': validation, 'test': test}

def retrieve_labels(dict):
    """
    Returns dictioanry of labels for each sample in the partion dictionary
    """
    labels = {}
    total_list = dict['train'] + dict['validation'] + dict['test']
    for enzyme in total_list:
        enzyme = enzyme[:4]
        csv_row = labeled_data.loc[labeled_data['label'] == enzyme]
        try:
            label = csv_row.type.iloc[0]
        except IndexError:
            raise ValueError('Information for {} doesnt exist in the provided labeled csv'.format(enzyme))
        labels[enzyme] = label
    return labels

### UNCOMMENT TO CREATE/POPULATE DIRECTORIES
build_populate_data()

### UNCOMMENT TO CREATE DICTIONARIES FOR GENERATOR CLAS
partition = retrieve_partioned_data('/data')
labels = retrieve_labels(partition)

### UNCOMMENT FOLLOWING BLOCK TO GENERATE NUMPY ARRAYS OF DATA INSTEAD OF DICTIONARY REFERENCES
# train_Xs, train_labels = retrieve_labeled_data("./data/train")
# validate_Xs, validate_labels = retrieve_labeled_data("./data/validation")
# test_Xs, test_labels = retrieve_labeled_data("./data/test")
