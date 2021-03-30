import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

def remove_excess_iso(x, f, batch_size, num_folds, num_cones, seed):
    """
    Removes some samples so dataset can be evenly separated into batches.
    Should only remove a small number of samples, i.e. for a batch size
    of 500 and 100223 samples, should choose 100000 samples.

    Parameters
    ----------
    x: Isolation cones
    f: Proportion labels
    batch_size: Learning batch size
    num_folds: Number of folds for kfold validation
    num_cones: Number of cones present in x
    seed: Numpy seed

    Returns
    -------
    x: Isolation cones remaining after selection
    f: Proportion labels corresponding to remaining cones
    """

    # Get counts for each label
    l, c = np.unique(f,return_counts=True)
    print(f'Labels: {l}, Initial Respective Counts {c}')

    # Get final counts
    neg_count = c[0]
    pos_count = c[1]
    neg_set_count = c[0] - (c[0] % (batch_size*num_folds))
    pos_set_count = c[1] - (c[1] % (batch_size*num_folds))

    # Randomly select samples
    np.random.seed(seed)
    indices_pos = np.random.choice(np.arange(0,pos_count-1),
            size=pos_set_count,replace=False)
    np.random.seed(seed)
    indices_neg = np.random.choice(np.arange(pos_count,len(f)),
            size=neg_set_count,replace=False)

    # Separate samples from the majority positive, majority negative sets
    samples_pos = x[indices_pos]
    labels_pos = f[indices_pos]
    samples_neg = x[indices_neg]
    labels_neg = f[indices_neg]

    # Allocate memory
    x = np.empty((pos_set_count+neg_set_count,18))
    f = np.empty(pos_set_count+neg_set_count)

    # Fill arrays
    x[:pos_set_count] = samples_pos
    x[pos_set_count:] = samples_neg
    f[:pos_set_count] = labels_pos
    f[pos_set_count:] = labels_neg

    l, c = np.unique(f,return_counts=True)
    print(f'Labels: {l}, Final Respective Counts {c}')

    return x, f

def shuffle_samples(x, f, batch_size, seed):
    """
    Shuffle samples maintaining batch size.

    Parameters
    ----------
    x: Samples
    f: Proportion labels
    batch_size: Learning batch size
    seed: Numpy seed

    Returns
    -------
    x: Shuffled samples
    f: Shuffled labels
    shuffle: Shuffle indices
    """

    shuffle = np.arange(len(x))
    shuffle = shuffle.reshape(-1, batch_size)
    np.random.seed(seed)
    np.random.shuffle(shuffle)
    shuffle = shuffle.flatten()
    x = x[shuffle]
    f = f[shuffle]

    return x, f, shuffle

def reshape_images_pytorch(x):
    """
    Reshape images for pytorch.

    Parameters
    ----------
    x: Images (4 dimensional)

    Returns
    -------
    x: Reshaped images
    """

    x = np.transpose(x,(0,2,3,1))

    return x

def make_llp_boolean_labels(f):
    """
    Make boolean labels corresponding to proportional labels, i.e.
    proportional labels 0.2, 0.8 -> 0, 1

    Parameters
    ----------
    f: Proportional labels

    Returns
    -------
    y: Binary labels
    """

    y = np.array([1 if i > 0.5 else 0 for i in f])

    return y

def select_cone_set(x, num_cones):
    """
    Select number of cones, grabbing cones from largest to smallest.

    Parameters
    ----------
    x: Isolation cones
    num_cones: Number of cones to grab

    Returns
    -------
    x: Chosen isolations
    """

    if num_cones < x.shape[-1]:
        cone_index = x.shape[-1] - num_cones - 1
    else:
        cone_index = None

    return x[:,:cone_index:-1]

def select_single_cone(x, cone_index):
    """
    Select single cone

    Parameters
    ----------
    x: Isolations
    cone_index: Index of cone to choose

    Returns
    -------
    x: Selected isolation
    """

    return x[:,cone_index:cone_index+1]

def iso_standard_scaler(train_dict, valid_dict, test_dict):
    """
    Use sklearn StandardScaler to scale isolations

    Parameters
    ----------
    train_dict: Dict containing train data
    valid_dict: Dict containing valid data
    test_dict: Dict containing test data

    Returns:
    train_dict: Dict containing scaled train data
    valid_dict: Dict containing scaled valid data
    test_dict: Dict containing scaled test data
    """

    scaler = StandardScaler()
    scaler.fit(train_dict['x'])
    train_dict['x'] = scaler.transform(train_dict['x'])
    valid_dict['x'] = scaler.transform(valid_dict['x'])
    test_dict['x'] = scaler.transform(test_dict['x'])

    return train_dict, valid_dict, test_dict

def make_llp_iso_data_loaders(train_dict, valid_dict, test_dict, 
                              batch_size=100, seed=123):
    """
    Make pytorch data loaders.

    Parameters
    ----------
    train_dict: Dict containing train data, labels, proportions
    valid_dict: Dict containing valid data, labels, proportions
    test_dict: Dict containing test data, labels, proportions, masses
    seed: numpy seed

    Returns
    -------
    data_loaders: Dict of pytorch train, valid, test dataloaders 
    """

    train_dict['x'] = torch.from_numpy(train_dict['x']).float()
    train_dict['y'] = torch.from_numpy(train_dict['y']).float()
    train_dict['f'] = torch.from_numpy(train_dict['f']).float()
    valid_dict['x'] = torch.from_numpy(valid_dict['x']).float()
    valid_dict['y'] = torch.from_numpy(valid_dict['y']).float()
    valid_dict['f'] = torch.from_numpy(valid_dict['f']).float()
    test_dict['x'] = torch.from_numpy(test_dict['x']).float()
    test_dict['y'] = torch.from_numpy(test_dict['y']).float()
    test_dict['f'] = torch.from_numpy(test_dict['f']).float()
    test_dict['m'] = torch.from_numpy(test_dict['m']).float()
    train_dataset = torch.utils.data.TensorDataset(train_dict['x'],
            train_dict['y'], train_dict['f'])
    valid_dataset = torch.utils.data.TensorDataset(valid_dict['x'],
            valid_dict['y'], valid_dict['f'])
    test_dataset = torch.utils.data.TensorDataset(test_dict['x'],
            test_dict['y'], test_dict['f'], test_dict['m'])

    # Create data loaders
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    data_loaders = {'train': DataLoader(dataset=train_dataset,
        batch_size=batch_size, shuffle=False, worker_init_fn=_init_fn), 
        'valid': DataLoader(dataset=valid_dataset, batch_size=batch_size,
            shuffle=False, worker_init_fn=_init_fn),
        'test': DataLoader(dataset=test_dataset, batch_size=batch_size,
            shuffle=False, worker_init_fn=_init_fn)}

    return data_loaders
