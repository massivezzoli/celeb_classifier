from utils.preprocessing import *

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

def dense_to_one_hot(labels):
    """Convert class labels from scalars to one-hot vectors.
    Parameters
    ----------
    labels : array
        Input labels to convert to one-hot representation.
    n_classes : int, optional
        Number of possible one-hot.
    Returns
    -------
    one_hot : array
        One hot representation of input.
    """
    # return np.eye(n_classes).astype(np.float32)[labels]
    return pd.get_dummies(labels).as_matrix()


class Dataset(object):
    """Create a dataset from data and their labels.
    Allows easy use of train/valid/test splits; Batch generator.
    Attributes
    ----------
    all_idxs : list
        All indexes across all splits.
    all_inputs : list
        All inputs across all splits.
    all_labels : list
        All labels across all splits.
    n_labels : int
        Number of labels.
    split : list
        Percentage split of train, valid, test sets.
    test_idxs : list
        Indexes of the test split.
    train_idxs : list
        Indexes of the train split.
    valid_idxs : list
        Indexes of the valid split.
    """

    def __init__(self, Xs, ys=None, zs=None, split=[1.0, 0.0, 0.0], one_hot=False, rnd_seed=1):
        """Initialize a Dataset object.
        Parameters
        ----------
        Xs : np.ndarray
            Images/inputs to a network
        ys : np.ndarray
            Labels/outputs to a network
        split : list, optional
            Percentage of train, valid, and test sets.
        one_hot : bool, optional
            Whether or not to use one-hot encoding of labels (ys).
        """
        self.all_idxs = []
        self.all_labels = []
        self.all_inputs = []
        self.train_idxs = []
        self.valid_idxs = []
        self.test_idxs = []
        self.n_labels = 0
        self.split = split
        self.rand_idxs_peek = []

        # Now mix all the labels that are currently stored as blocks
        self.all_inputs = Xs
        n_idxs = len(self.all_inputs)
        idxs = range(n_idxs)
        if rnd_seed:
            np.random.seed(rnd_seed)
        rand_idxs = np.random.permutation(idxs)
        self.rand_idxs_peek = rand_idxs
        self.all_inputs = self.all_inputs[rand_idxs, ...]
        if ys is not None:
            self.all_labels = ys if not one_hot else dense_to_one_hot(ys)
            self.all_labels = self.all_labels[rand_idxs, ...]
        else:
            self.all_labels = None
        if zs is not None:
            self.all_bottles = zs
            self.all_bottles = self.all_bottles[rand_idxs, ...]
        else:
            self.all_bottles = None

        # Get splits
        self.train_idxs = idxs[:round(split[0] * n_idxs)]
        self.valid_idxs = idxs[len(self.train_idxs):
                               len(self.train_idxs) + round(split[1] * n_idxs)]
        self.test_idxs = idxs[
            (len(self.valid_idxs) + len(self.train_idxs)):
            (len(self.valid_idxs) + len(self.train_idxs)) +
            round(split[2] * n_idxs)]

        # TODO: Suppose bottenecks exist
        #self.bottle_mean_array = self._bottle_means(one_hot)

    @property
    def X(self):
        """Inputs/Xs/Images.
        Returns
        -------
        all_inputs : np.ndarray
            Original Inputs/Xs.
        """
        return self.all_inputs

    @property
    def Y(self):
        """Outputs/ys/Labels.
        Returns
        -------
        all_labels : np.ndarray
            Original Outputs/ys.
        """
        return self.all_labels

    @property
    def Z(self):
        """Outputs/ys/Labels.
        Returns
        -------
        all_labels : np.ndarray
            Original Outputs/ys.
        """
        return self.all_bottles
    
    @property
    def train(self):
        """Train split.
        Returns
        -------
        split : DatasetSplit
            Split of the train dataset.
        """
        if len(self.train_idxs):
            inputs = self.all_inputs[self.train_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.train_idxs, ...]
            else:
                labels = None
            if self.all_bottles is not None:
                bottles = self.all_bottles[self.train_idxs, ...]
            else:
                bottles = None
        else:
            inputs, labels, bottles = [], [], []
        return DatasetSplit(inputs, labels, bottles)

    @property
    def valid(self):
        """Validation split.
        Returns
        -------
        split : DatasetSplit
            Split of the validation dataset.
        """
        if len(self.valid_idxs):
            inputs = self.all_inputs[self.valid_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.valid_idxs, ...]
            else:
                labels = None
            if self.all_bottles is not None:
                bottles = self.all_bottles[self.valid_idxs, ...]
            else:
                bottles = None
        else:
            inputs, labels, bottles = [], [], []
        return DatasetSplit(inputs, labels, bottles)

    @property
    def test(self):
        """Test split.
        Returns
        -------
        split : DatasetSplit
            Split of the test dataset.
        """
        if len(self.test_idxs):
            inputs = self.all_inputs[self.test_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.test_idxs, ...]
            else:
                labels = None
            if self.all_bottles is not None:
                bottles = self.all_bottles[self.test_idxs, ...]
            else:
                bottles = None
        else:
            inputs, labels, bottles = [], [], []
        return DatasetSplit(inputs, labels, bottles)

    def mean(self):
        """Mean of the inputs/Xs.
        Returns
        -------
        mean : np.ndarray
            Calculates mean across 0th (batch) dimension.
        """
        return np.mean(self.all_inputs, axis=0)

    def std(self):
        """Standard deviation of the inputs/Xs.
        Returns
        -------
        std : np.ndarray
            Calculates std across 0th (batch) dimension.
        """
        return np.std(self.all_inputs, axis=0)

    # TODO: Suppose bottenecks exist
    #def bottle_means(self):
    #    return self.bottle_mean_array
    # TODO: Suppose bottenecks exist
    #def _bottle_means(self, one_hot):
    #    target_list = []
    #    train_labels = self.all_labels[self.train_idxs, ...]
    #    this_all_labels = self.all_labels
    #    #n_labels = train_labels.shape[1]
    #    n_labels =len(set(train_labels))
    #    if one_hot:
    #        train_label_vals = np.argmax(train_labels, axis=1)
    #    else:
    #        train_label_vals = train_labels
    #    #train_label_vals = np.argmax(train_labels, axis=1)
    #    for target in range(n_labels):
    #        target_idx = np.where(train_label_vals == target)
    #        target_list.append(np.mean(self.all_bottles[target_idx], axis=0))
    #    bottle_array = np.array(target_list)
    #    np.savetxt('bottle_means.txt', bottle_array)
    #    return bottle_array


class DatasetSplit(object):

    #def __init__(self, images, labels):
    def __init__(self, images, labels, bottles, rnd_seed=1):
        self.images = np.array(images).astype(np.float32)
        if labels is not None:
            self.labels = np.array(labels).astype(np.int32)
            self.n_labels = len(np.unique(labels))
        else:
            self.labels = None
            
        if bottles is not None:
            self.bottles = np.array(bottles).astype(np.float64)
        else:
            self.bottles = None
            
        self.num_examples = len(self.images)
        self.rnd_seed = rnd_seed

    def next_batch(self, batch_size=100):
        # Shuffle each epoch
        if self.rnd_seed:
            np.random.seed(self.rnd_seed)
        current_permutation = np.random.permutation(range(len(self.images)))
        epoch_images = self.images[current_permutation, ...]
        if self.labels is not None:
            epoch_labels = self.labels[current_permutation, ...]
        if self.bottles is not None:
            epoch_bottles = self.bottles[current_permutation, ...]            

        # Then iterate over the epoch
        self.current_batch_idx = 0
        while self.current_batch_idx < len(self.images):
            end_idx = min(
                self.current_batch_idx + batch_size, len(self.images))
            this_batch = {
                'images': epoch_images[self.current_batch_idx:end_idx],
                'labels': epoch_labels[self.current_batch_idx:end_idx] if self.labels is not None else None,
                'bottles': epoch_bottles[self.current_batch_idx:end_idx] if self.bottles is not None else None
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels'], this_batch['bottles']
