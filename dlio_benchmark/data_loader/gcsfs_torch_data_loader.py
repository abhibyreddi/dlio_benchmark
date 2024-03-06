"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from time import time
import logging
import math
import os
import io
import pickle
import torch
import gcsfs
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType, FormatType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.config import ConfigArguments
from dlio_profiler.logger import fn_interceptor as Profile

from pydicom import dcmread

dlp = Profile(MODULE_DATA_LOADER)


class GCSFSTorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch

        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.dlp_logger = None
        if num_workers == 0:
            self.worker_init(-1)

        self.gcp_project_name = args.gcp_project_name
        # Initialize GCSFS
        self.gcs_fs = gcsfs.GCSFileSystem(
            project=self.gcp_project_name,         
            access='read_only', 
        )
        # List all files in the dataset
        prefix = args.data_folder
        if self.dataset_type == DatasetType.TRAIN:
            prefix = os.path.join(prefix,  "train")
        elif self.dataset_type == DatasetType.VALID:
            prefix = os.path.join(prefix, "valid")
        dataset = os.path.join(args.gcs_bucket, prefix)
        logging.info(f"Listing files in {dataset} with GCSFS")
        self.files = self.gcs_fs.ls(dataset)
        logging.info(f"Found {len(self.files)} files")
        self.num_samples = len(self.files)

        # Initialize reader function
        if format_type == FormatType.NPZ:
            self.format_fn = lambda b: np.load(io.BytesIO(b), allow_pickle=True)["x"]
        elif format_type == FormatType.DCM:
            def parse_dcm(b):
                a = dcmread(io.BytesIO(b)).pixel_array
                logging.debug(f"Read dcm image. Size: {a.size}; Type: {a.dtype}")
                return torch.rand((512, 512))
            self.format_fn = parse_dcm
        else:
            self.format_fn = lambda b: b

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()
    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, image_idx):
        fs = gcsfs.GCSFileSystem(
            project=self.gcp_project_name,         
            access='read_only',
        )
        with fs.open(self.files[image_idx], 'rb') as f:
            return self.format_fn(f.read())

class GCSFSTorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

    @dlp.log
    def read(self):
        dataset = GCSFSTorchDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples, self._args.read_threads, self.batch_size)
        if self._args.sample_shuffle != Shuffle.OFF:
            # torch seed is used for all functions within.
            torch.manual_seed(self._args.seed)
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            # generator needs to load up torch seed.
            torch_generator = torch.Generator()
            torch_generator.manual_seed(seed)
            # Pass generator to sampler
            sampler = RandomSampler(dataset, generator=torch_generator)
        else:
            sampler = SequentialSampler(dataset)
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                logging.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        logging.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers {torch.__version__}")
        if self._args.read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            if torch.__version__ != '1.3.1':       
                kwargs['persistent_workers'] = True
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True,
                                       **kwargs)
        else: 
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True,
                                       **kwargs)  # 2 is the default value
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        for batch in self._dataset:
            yield batch

    @dlp.log
    def finalize(self):
        pass
