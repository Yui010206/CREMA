"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import gzip
import logging
import cv2

import random as rnd
import tarfile
import zipfile

import decord
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset, ChainDataset
from decord import VideoReader
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import ConcatDataset
from tqdm import tqdm

from PIL import Image

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")


def readFlow(filename, new_size=(256, 256)):
    """ Read optical flow from file.
    
    The function reads optical flow from a .flo file and returns it as a numpy array.
    
    Args:
    filename (str): The path to the .flo file.
    
    Returns:
    numpy.ndarray: A numpy array containing the optical flow with u and v components stacked in depth.
    """
    # Open the file in binary mode
    with open(filename, 'rb') as f:
        # Check the magic number in the header matches the expected .flo format
        magic = np.fromfile(f, np.float32, count=1)[0]
        assert magic == 202021.25, 'Invalid .flo file'

        # Read the width and height of the flow field
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # Read the flow field data
        flow_data = np.fromfile(f, np.float32, count=2 * width * height)

        # Reshape the flow field data into a 3D numpy array with shape (height, width, 2)
        flow = np.reshape(flow_data, (height, width, 2))
        flow = np.transpose(flow, (1, 0, 2))

    flow_resized = cv2.resize(flow, new_size[::-1], interpolation=cv2.INTER_LINEAR)
    # min_val = flow_resized.min()
    # max_val = flow_resized.max()
    # flow_resized = 2 * (flow_resized - min_val) / (max_val - min_val) - 1

    return flow_resized

def load_flow(frames_dir, indices,
               height=-1, width=-1):

    #print(frames_dir)
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.split('.')[-1] in ['flo']])
    frms = []
    for idx in indices:
        # print(idx, frame_files[idx])
        flow = readFlow(frame_files[idx], (height, width))
        min_val = flow.min()
        max_val = flow.max()
        flow = 2 * (flow - min_val) / (max_val - min_val) - 1
        frms.append(np.asarray(flow))
    
    frms = np.stack(frms).astype(np.float32)  # (C, T, H, W)
    # print('frms',frms.shape)
    # print(frms.shape)
    frms = torch.from_numpy(frms) #.unsqueeze(1)  # (T, 1, H, W)
    frms = frms.permute(0,3,1,2)
    return frms

def load_depth(frames_dir, indices,
               height=-1, width=-1, invalid_val=-99):
    
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.split('.')[-1] in ['jpg', 'png']])
    # print(frame_files)
    # print(len(indices), indices)
    frms = []
    for idx in indices:
        depth = Image.open(frame_files[idx])
        depth = depth.resize((width, height))
        depth = np.asarray(depth)

        invalid_mask = depth == invalid_val
        mask = np.logical_not(invalid_mask)
        vmin = np.percentile(depth[mask],2) 
        vmax = np.percentile(depth[mask],85)
        if vmin != vmax:
            depth = (depth - vmin) / (vmax - vmin) 
        else:
            depth = depth * 0.

        frms.append(np.asarray(depth))
    
    frms = np.stack(frms).astype(np.float32)  # (C, T, H, W)
    frms = torch.from_numpy(frms).unsqueeze(1)  # (1, C, T, H, W)

    return frms

def load_frames(frames_dir, n_frms=MAX_INT, height=-1, width=-1, 
                sampling="uniform", clip_proposal=None, type='rgb', indices=None):
    
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.split('.')[-1] in ['jpg', 'png']])
    # print('frame_files', len(frame_files))
    vlen = len(frame_files) - 1 # for flow
    n_frms = n_frms #min(n_frms, vlen)
    
    if clip_proposal is None:
        start, end = 0, vlen
    else:
        start, end = int(clip_proposal[0]*vlen), int(clip_proposal[1]*vlen)
        if start < 0:
            start = 0
        if end > vlen:
            end = vlen

    intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
    ranges = [(intervals[i], intervals[i + 1]) for i in range(len(intervals) - 1)]
    
    if indices is None:
        if sampling == 'random':
            indices_ = [rnd.choice(range(x[0], x[1])) if x[0] != x[1] else x[0] for x in ranges]
        elif sampling == 'uniform':
            indices_ = [(x[0] + x[1]) // 2 for x in ranges]
        # elif sampling == "headtail":
        #     indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        #     indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        #     indices = indices_h + indices_t
        else:
            raise NotImplementedError
    else:
        indices_ = indices
    
    # print(len(indices_) , n_frms, indices_)

    if len(indices_) < n_frms:
        extra = [indices_[-1] for i in range(n_frms - len(indices_))]
        indices_.extend(extra)
    frms = []
    # debug = [frame_files[idx] for idx in indices]
    # print(debug)
    # print('frame_files',len(frame_files))
    # print('indices_', indices_)
    for idx in indices_:
        # print(frame_files[idx])
        with Image.open(frame_files[idx]) as img:
            if type == 'norm':
                img = img.convert('RGB')
            if type == 'flow':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = img.transpose(2) # ROTATE_90
                # print('here')
            if type == 'depth':
                 img = img.convert('RGB')
                 
            if height > 0 and width > 0:
                img = img.resize((width, height))
            frms.append(np.asarray(img))
    
    frms = np.stack(frms).transpose(3, 0, 1, 2).astype(np.float32)  # (C, T, H, W)
    frms = torch.from_numpy(frms)
    # print('frms', frms.shape) # 3, 4, 224, 224 for rgb,  4, 4, 224, 224 for norm, 3, 4, 224, 224 for norm.convert('RGB')

    return frms, indices_

# add for loading video
def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", clip_proposal=None):
    vr = VideoReader(uri=video_path, height=height, width=width)
    vlen = len(vr)
    n_frms = min(n_frms, vlen)
    fps = vr.get_avg_fps() 
    if clip_proposal is None:
        start, end = 0, vlen
    else:
        start, end = int(clip_proposal[0]*fps), int(clip_proposal[1]*fps)
        if start < 0:
            start = 0
        if end > vlen:
            end = vlen

    intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))

    if sampling == 'random':
        indices = []
        for x in ranges:
            if x[0] == x[1]:
                indices.append(x[0])
            else:
                indices.append(rnd.choice(range(x[0], x[1])))
    elif sampling == 'uniform':
        
        indices = [(x[0] + x[1]) // 2 for x in ranges]

    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError
    
    if len(indices) < n_frms:
        rest = [indices[-1] for i in range(n_frms - len(indices))]
        indices = indices + rest 
    # get_batch -> T, H, W, C
    frms = vr.get_batch(indices).permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms, indices, fps

def load_video_demo(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", clip_proposal=None):
    vr = VideoReader(uri=video_path, height=height, width=width)
    vlen = len(vr)
    n_frms = min(n_frms, vlen)
    fps = vr.get_avg_fps() 
    if clip_proposal is None:
        start, end = 0, vlen
    else:
        start, end = int(clip_proposal[0]*fps), int(clip_proposal[1]*fps)
        if start < 0:
            start = 0
        if end > vlen:
            end = vlen

    intervals = np.linspace(start=start, stop=end, num=n_frms + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))

    if sampling == 'random':
        indices = []
        for x in ranges:
            if x[0] == x[1]:
                indices.append(x[0])
            else:
                indices.append(rnd.choice(range(x[0], x[1])))
    elif sampling == 'uniform':
        
        indices = [(x[0] + x[1]) // 2 for x in ranges]

    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError
    
    if len(indices) < n_frms:
        rest = [indices[-1] for i in range(n_frms - len(indices))]
        indices = indices + rest 
    # get_batch -> T, H, W, C
    
    frms = vr.get_batch(indices)
    frms = frms.asnumpy()
    frms = torch.from_numpy(frms)
    frms = frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    return frms, indices, fps, vlen

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.

    Args:
        from_path: the path of the archive.
        to_path: the root path of the extracted files (directory of from_path)
        overwrite: overwrite existing files (False)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
        >>> ['.data/val.de', '.data/val.en']

    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {} to {}.".format(from_path, to_path))
        with tarfile.open(from_path, "r") as tar:
            files = []
            for file_ in tqdm(tar):
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            logging.info("Finished extracting tar file {}.".format(from_path))
            return files

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {} to {}.".format(from_path, to_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = []
            for file_ in tqdm(zfile.namelist()):
                file_path = os.path.join(to_path, file_)
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        files = [f for f in files if os.path.isfile(f)]
        logging.info("Finished extracting zip file {}.".format(from_path))
        return files

    elif from_path.endswith(".gz"):
        logging.info("Opening gz file {} to {}.".format(from_path, to_path))
        default_block_size = 65536
        filename = from_path[:-3]
        files = [filename]
        with gzip.open(from_path, "rb") as gzfile, open(filename, "wb") as d_file:
            while True:
                block = gzfile.read(default_block_size)
                if not block:
                    break
                else:
                    d_file.write(block)
            d_file.write(block)
        logging.info("Finished extracting gz file {}.".format(from_path))
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz, .gz and zip achives."
        )


def save_frames_grid(img_array, out_path):
    import torch
    from PIL import Image
    from torchvision.utils import make_grid

    if len(img_array.shape) == 3:
        img_array = img_array.unsqueeze(0)
    elif len(img_array.shape) == 5:
        b, t, c, h, w = img_array.shape
        img_array = img_array.view(-1, c, h, w)
    elif len(img_array.shape) == 4:
        pass
    else:
        raise NotImplementedError(
            "Supports only (b,t,c,h,w)-shaped inputs. First two dimensions can be ignored."
        )

    assert img_array.shape[1] == 3, "Exepcting input shape of (H, W, 3), i.e. RGB-only."

    grid = make_grid(img_array)
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    img = Image.fromarray(ndarr)

    img.save(out_path)
