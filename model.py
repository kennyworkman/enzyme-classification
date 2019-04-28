import os
import numpy as np
import pandas as pd

labeled_data = pd.read_csv("Xs_generator/batch_classification.csv", sep='\t')

def retrieve_Xs(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.npy')]

def retrieve_labels(Xs):
    """
    Need to pass a list of PATHS for each data file
    """
    for path in Xs:
        filename = os.path.basename(path)
        print(labeled_data.loc(labeled_data['label'] == filename[:4]))

Xs = retrieve_Xs('/Users/kenny/desktop/chem195/enzyme_classifier/Xs_generator/Xs')

retrieve_labels(Xs)
