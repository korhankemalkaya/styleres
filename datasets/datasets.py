# python3.7
"""Contains the class of dataset."""

import os
import pickle
import string
import zipfile
import numpy as np
import cv2
import lmdb
from PIL import Image, ImageDraw
import math

import torch
from torch.utils.data import Dataset

from .transforms import progressive_resize_image
from .transforms import crop_resize_image
from .transforms import resize_image
from .transforms import normalize_image

try:
    import turbojpeg
    BASE_DIR = os.path.dirname(os.path.relpath(__file__))
    LIBRARY_NAME = 'libturbojpeg.so.0'
    LIBRARY_PATH = os.path.join(BASE_DIR, LIBRARY_NAME)
    jpeg = turbojpeg.TurboJPEG(LIBRARY_PATH)
except ImportError:
    jpeg = None

__all__ = ['BaseDataset']

_FORMATS_ALLOWED = ['dir', 'lmdb', 'list', 'zip']


class ZipLoader(object):
    """Defines a class to load zip file.

    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


class LmdbLoader(object):
    """Defines a class to load lmdb file.

    This is a static class, which is used to solve lmdb loading error
    when num_workers > 0
    """
    files = dict()

    @staticmethod
    def get_lmdbfile(file_path):
        """Fetches a lmdb file"""
        lmdb_files = LmdbLoader.files
        if 'env' not in lmdb_files:
            env = lmdb.open(file_path,
                            max_readers=1,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                num_samples = txn.stat()['entries']
            cache_file = '_cache_' + ''.join(
                c for c in file_path if c in string.ascii_letters)
            if os.path.isfile(cache_file):
                keys = pickle.load(open(cache_file, "rb"))
            else:
                with env.begin(write=False) as txn:
                    keys = [key for key, _ in txn.cursor()]
                pickle.dump(keys, open(cache_file, "wb"))
            lmdb_files['env'] = env
            lmdb_files['num_samples'] = num_samples
            lmdb_files['keys'] = keys
        return lmdb_files

    @staticmethod
    def get_image(file_path, idx):
        """Decodes an image from a particular lmdb file"""
        lmdb_files = LmdbLoader.get_lmdbfile(file_path)
        env = lmdb_files['env']
        keys = lmdb_files['keys']
        with env.begin(write=False) as txn:
            imagebuf = txn.get(keys[idx])
        image_np = np.frombuffer(imagebuf, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


class BaseDataset(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """
    def __init__(self,
                 root_dir,
                 resolution,
                 data_format='dir',
                 image_list_path=None,
                 mirror=0.0,
                 progressive_resize=True,
                 crop_resize_resolution=-1,
                 transform=normalize_image,
                 masked_ratio =[0.0,1.0],
                 transform_kwargs=None,
                 ann_dir = None,
                 attribute_tag=None,
                 ispositive=False,
                 **_unused_kwargs):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            resolution: The resolution of the returned image.
            data_format: Format the dataset is stored. Supports `dir`, `lmdb`,
                and `list`. (default: `dir`)
            image_list_path: Path to the image list. This field is required if
                `data_format` is `list`. (default: None)
            mirror: The probability to do mirror augmentation. (default: 0.0)
            progressive_resize: Whether to resize images progressively.
                (default: True)
            crop_resize_resolution: The resolution of the output after crop
                and resize. (default: -1)
            masked_ratio: The ratio of the masked region with respect to the image. [0.0, 0.0] for no mask
            transform: The transform function for pre-processing.
                (default: `datasets.transforms.normalize_image()`)
            transform_kwargs: The additional arguments for the `transform`
                function. (default: None)

        Raises:
            ValueError: If the input `data_format` is not supported.
            NotImplementedError: If the input `data_format` is not implemented.
        """
        if data_format.lower() not in _FORMATS_ALLOWED:
            raise ValueError(f'Invalid data format `{data_format}`!\n'
                             f'Supported formats: {_FORMATS_ALLOWED}.')

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_format = data_format.lower()
        self.image_list_path = image_list_path
        self.mirror = np.clip(mirror, 0.0, 1.0)
        self.progressive_resize = progressive_resize
        self.crop_resize_resolution = crop_resize_resolution
        self.transform = transform
        self.transform_kwargs = transform_kwargs or dict()
        self.masked_ratio = masked_ratio
        self.attribute_tag = attribute_tag

        if self.data_format == 'dir':
            if attribute_tag is None:
                self.image_paths = sorted(os.listdir(self.root_dir))
            elif attribute_tag == 'smile':
                compare = "1" if ispositive else "-1"
                smile_idx = 32
                ann_data = np.loadtxt(ann_dir, skiprows=2, dtype=str, usecols=(0,smile_idx))
                smile_array = ann_data[ ann_data[:,1] == compare]
                self.image_paths = list (smile_array[:,0])
            self.num_samples = len(self.image_paths)
        elif self.data_format == 'lmdb':
            lmdb_file = LmdbLoader.get_lmdbfile(self.root_dir)
            self.num_samples = lmdb_file['num_samples']
        elif self.data_format == 'list':
            self.metas = []
            assert os.path.isfile(self.image_list_path)
            with open(self.image_list_path) as f:
                for line in f:
                    fields = line.rstrip().split(' ')
                    if len(fields) == 1:
                        self.metas.append((fields[0], None))
                    else:
                        assert len(fields) == 2
                        self.metas.append((fields[0], int(fields[1])))
            self.num_samples = len(self.metas)
        elif self.data_format == 'zip':
            zip_file = ZipLoader.get_zipfile(self.root_dir)
            image_paths = [f for f in zip_file.namelist()
                           if ('.jpg' in f or '.jpeg' in f or '.png' in f)]
            self.image_paths = sorted(image_paths)
            self.num_samples = len(self.image_paths)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = dict()

        # Load data.
        if self.data_format == 'dir':
            image_path = self.image_paths[idx]
            # try:
            #     in_file = open(os.path.join(self.root_dir, image_path), 'rb')
            #     image = jpeg.decode(in_file.read())
            # except:  # pylint: disable=bare-except
            image = cv2.imread(os.path.join(self.root_dir, image_path))
        elif self.data_format == 'lmdb':
            image = LmdbLoader.get_image(self.root_dir, idx)
        elif self.data_format == 'list':
            image_path, label = self.metas[idx]
            image = cv2.imread(os.path.join(self.root_dir, image_path))
            label = None if label is None else torch.LongTensor(label)
            # data.update({'label': label})
        elif self.data_format == 'zip':
            image_path = self.image_paths[idx]
            image = ZipLoader.get_image(self.root_dir, image_path)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

        image = image[:, :, ::-1]  # Converts BGR (cv2) to RGB.

        # Transform image.
        if self.crop_resize_resolution > 0:
            image = crop_resize_image(image, self.crop_resize_resolution)
        if self.progressive_resize:
            image = progressive_resize_image(image, self.resolution)
        image = image.transpose(2, 0, 1).astype(np.float32)
        if np.random.uniform() < self.mirror:
            image = image[:, :, ::-1]  # CHW
        image = torch.FloatTensor(image.copy())
        if not self.progressive_resize:
            image = resize_image(image, self.resolution)

        if self.transform is not None:
            image = self.transform(image, **self.transform_kwargs)
        data.update({'image': image})
        data.update({'name': image_path})

        # if (self.masked_ratio[0] == 0.0 and self.masked_ratio[1] == 0.0):
        #     valid = np.ones(shape=(1, self.resolution, self.resolution)).astype('float32') #All pixels are valid.
        # else:
        #     valid = self.RandomMask(self.resolution, self.masked_ratio)
        # data.update({'valid': valid})

        return data

    def RandomBrush(self,
                    max_tries,
                    s,
                    min_num_vertex = 4,
                    max_num_vertex = 18,
                    mean_angle = 2*math.pi / 5,
                    angle_range = 2*math.pi / 15,
                    min_width = 12,
                    max_width = 48):
        H, W = s, s
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def RandomMask(self, s, hole_range=[0.0,1.0]):
        coef = min(hole_range[0] + hole_range[1], 1.0)
        while True:
            mask = np.ones((s, s), np.uint8)
            def Fill(max_size):
                w, h = np.random.randint(max_size), np.random.randint(max_size)
                ww, hh = w // 2, h // 2
                x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
                mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
            def MultiFill(max_tries, max_size):
                for _ in range(np.random.randint(max_tries)):
                    Fill(max_size)
            #MultiFill(int(10 * coef), s // 2)
            MultiFill(int(4 * coef), s)
            mask = np.logical_and(mask, 1 - self.RandomBrush(int(15 * coef), s))
            hole_ratio = 1 - np.mean(mask)
            if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
                continue
            return mask[np.newaxis, ...].astype(np.float32)