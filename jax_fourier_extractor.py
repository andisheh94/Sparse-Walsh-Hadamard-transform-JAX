from swht_jax.swht_jax import get_time_samples, setup_fourier
import tqdm
import jax.numpy as jnp
# https://pypi.org/project/hadamard-transform/
from hadamard_transform import hadamard_transform
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import numpy as np
import utils, models
import pathlib
import pickle
#use float 64 tensors
from jax.config import config
config.update("jax_enable_x64", True)
from fourier_extractor.jax_fourier_wrapper import FourierWrapper
import os
import time

class SparseFourierDataset(torch.utils.data.Dataset):
    def __init__(self, time_samples):
        # T * n * 2^b array
        time_samples = np.array(time_samples)
        T, n, B = time_samples.shape
        self.T, self.n, self.B = T, n, B
        shifted_samples = []
        for i in range(T):
            shifts = np.eye(self.n, dtype=np.int32).reshape((n, n, 1))
            # no_shifts * n * 2^b array
            time_samples_with_shifts = (np.tile(np.array(time_samples[i]), reps=(n, 1, 1)) + shifts) % 2
            shifted_samples.append(time_samples_with_shifts)
        # T * no_shifts * n * 2^b array
        shifted_samples = np.array(shifted_samples)
        # T * n * (no_shifts*2^b array)
        all_samples = [
            np.concatenate([time_samples[i], np.swapaxes(shifted_samples[i], 0, 1).reshape((n, n * B))], axis=1)
            for i in range(T)]
        # T * n * (no_shifts*2^b array)
        all_samples = np.array(all_samples)
        # n * (T * no_shifts * 2^b) array
        all_samples = np.swapaxes(all_samples, 0, 1).reshape((n, -1))
        self.all_samples = all_samples.T
        self.no_samples = T * B * (n + 1)

    def __len__(self):
        return self.no_samples

    # get a row at an index
    def __getitem__(self, idx):
        return self.all_samples[idx]


def sample_and_wht(f, n, b, T, batch_size, num_workers, model_type):
    """
    f: callable to be sampled
    n: dimension of domain of f
    b: 2^b is no buckets
    T: no. of peeling rounds
    """
    # prepare time samples
    print("getting time samples")
    time_samples = get_time_samples(n, b, T)
    print("creating all time samples with shifts")
    dataset = SparseFourierDataset(time_samples)
    no_samples = len(dataset)
    print("creating dataloader")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # evaluate function on time samples
    y = torch.zeros((no_samples))
    if model_type == "nn":
        print("sampling neural network")
        torch.cuda.set_device("cuda:1")
        with torch.no_grad():
            for i, x in tqdm.tqdm(enumerate(dataloader)):
                x = x.cuda().double()
                evaluation = f(x).squeeze().cpu()
                y[i * batch_size:(i + 1) * batch_size] = evaluation
    else:
        print("sampling tree")
        for i, x in tqdm.tqdm(enumerate(dataloader)):
            x = np.array(x)
            evaluation = f.predict(x)
            y[i * batch_size:(i + 1) * batch_size] = torch.tensor(evaluation)

    y = y.reshape((T * (n + 1), -1))
    # Get WHT
    y_wht = torch.zeros(y.shape)
    print("getting wht")
    for i in tqdm.tqdm(range(y.shape[0])):
        y_wht[i] = hadamard_transform(y[i]) / np.power(2, b/2)
    ref_wht_index = slice(0, -1, n + 1)
    # convert from torch tensors to jax arrays
    print("converting WHTs to jax arrays")
    ref_wht = jnp.array(y_wht[ref_wht_index], dtype=jnp.float64)
    shifted_wht = jnp.array(np.delete(y_wht, ref_wht_index, axis=0), dtype=jnp.float64).reshape((T, n, -1))
    return ref_wht, shifted_wht


def save_to_cache(results, dataset, model, b, depth=""):
    this_directory = pathlib.Path(__file__).parent.resolve()
    if not os.path.exists(f"{this_directory}/cache_jax"):
        os.makedirs(f"{this_directory}/cache_jax/")
    path_to_cache = f"{this_directory}/cache_jax/{dataset}_{model}{depth}_b={b}.pkl"
    with open(path_to_cache, "wb") as file:
        pickle.dump(results, file)


def load_model(dataset, model, depth=None):
    if model == "nn":
        f = models.nn_model.load_model(dataset, best=True)
    elif model == "random_forest":
        f = models.random_forest_model.load_model(dataset, depth)
    elif model == "catboost":
        f = models.catboost_model.load_model(dataset, depth)
    return f


def compute_fourier(dataset, model, b, depth=None, save_result=True):
    try:
        return load_from_cache(dataset, model, b, depth)
    except:
        pass

    task_settings = utils.get_task_settings()
    # get run settings
    no_features = task_settings["no_features"][dataset]
    if model=="nn":
        batch_size = task_settings["batch_size_nn_sampling"][dataset]
    else: # catboost or random_forest
        batch_size = task_settings["batch_size_tree_sampling"][dataset]

    num_workers = task_settings["no_workers"][dataset]
    # hard-coded for now
    T = 5
    # load model
    f = load_model(dataset, model, depth)
    # get time samples needed by Fourier transform
    print(f"b={b} B={2 ** b}")
    # get compiled fourier transform function

    now = time.time()
    print("jitting sparse fourier")
    sparse_wht = setup_fourier(n=no_features, b=b, T=T)
    # evaluate neural net on those samples
    print("sampling")
    ref_whts, shifted_whts = sample_and_wht(f=f, n=no_features, b=b, T=T, batch_size=batch_size,
                                            num_workers=num_workers, model_type=model)
    print("computing sparse WHT")
    freqs, amps = sparse_wht(ref_whts, shifted_whts)
    then = time.time()
    fourier_compute_time = then - now
    if save_result:
        save_to_cache([freqs, amps, fourier_compute_time], dataset, model, b, depth)
    return freqs, amps, fourier_compute_time


class NNFourierDataset(Dataset):
    def __init__(self, time_samples):
        self.no_samples, self.no_features = time_samples.shape
        self.time_samples = np.array(time_samples, dtype=np.float64)
        pass
    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        return self.time_samples[idx]


def get_predictions(dataset, model, depth, time_samples):
    """ Compute model predictions evaluated on the time samples
    """
    f = load_model(dataset, model, depth)
    nn_fourier_dataset = NNFourierDataset(time_samples)
    if model == "nn":
        dataset_batch_size = utils.get_task_settings()["batch_size_nn_sampling"][dataset]
    else:
        dataset_batch_size = utils.get_task_settings()["batch_size_tree_sampling"][dataset]
    dataloader = DataLoader(nn_fourier_dataset, batch_size=dataset_batch_size, shuffle=False)
    predictions = []

    if model == "nn":
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.cuda()
                yhat = f(inputs).squeeze()
                yhat = yhat.cpu().numpy()
                predictions.append(yhat)

    else:
        for inputs in dataloader:
            inputs = np.array(inputs)
            yhat = f.predict(inputs)
            predictions.append(yhat)

    predictions = np.concatenate(predictions, axis=0)

    return predictions


def test_fourier_r2(fourier_transform, dataset, model, depth):
    x_train, x_test, y_train, y_test = utils.get_dataset(task_name=dataset, with_splits=True)
    # Random samples
    random_samples_shape = (1000, x_train.shape[1])
    x_random = np.random.randint(low=0, high=2, size=random_samples_shape)

    # Get predictions from Fourier
    pred_fourier_test = fourier_transform[x_test]
    pred_fourier_random = fourier_transform[x_random]

    # Get predictions from neural net
    pred_test = get_predictions(dataset, model, depth, x_test)
    pred_random = get_predictions(dataset, model, depth, x_random)

    metric_dict = {
        "fourier_test_r2": r2_score(pred_test, pred_fourier_test),
        "random_r2": r2_score(pred_random, pred_fourier_random)
    }

    return metric_dict


def load_from_cache(dataset, model, b, depth):

    this_directory = pathlib.Path(__file__).parent.resolve()
    path_to_cache = f"{this_directory}/cache_jax/{dataset}_{model}{depth}_b={b}.pkl"
    with open(path_to_cache, "rb") as file:
        results = pickle.load(file)
    return results


def test_fourier(dataset, model, depth):
    task_settings = utils.get_task_settings()
    # get run settings
    b_min, b_max = task_settings["b_range"][dataset]
    # hard-coded for now
    metrics = []
    # get time samples needed by Fourier transform
    for b in range(b_min, b_max + 1):
        print(f"b={b} B={2 ** b}")
        # get compiled fourier transform function
        try:
            r = load_from_cache(dataset, model, b, depth)
        except:
            break
        if len(r) == 2:
            freqs, amps = r
        else:
            freqs, amps, _ = r
        wrapped_fourier = FourierWrapper(dataset, b, freqs, amps)
        metric = test_fourier_r2(wrapped_fourier, dataset, model, depth)
        metrics.append((b, metric))
        print(f"Fourier quality b={b}:", metric)
    return metrics


if __name__ == "__main__":
    task_name = "sgemm"
    # task_name = "harvard60"
    compute_fourier(task_name)
    test_fourier(task_name)

