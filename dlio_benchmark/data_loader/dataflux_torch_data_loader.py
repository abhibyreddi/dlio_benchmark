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
import pickle
import torch
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
import dataflux_pytorch

from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.config import ConfigArguments
from dlio_profiler.logger import fn_interceptor as Profile

dlp = Profile(MODULE_DATA_LOADER)

class DatafluxTorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.DF_PYTORCH)
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=0,
                                               epoch_number=self.epoch_number)
                
    @dlp.log
    def read(self):
        prefix=self._args.data_folder
        if self.dataset_type == DatasetType.TRAIN:
            prefix = prefix + "/train"
        elif self.dataset_type == DatasetType.VALID:
            prefix = prefix + "/valid"
        format_fn = lambda b: np.load(io.BytesIO(b), allow_pickle=True)["x"]
        df_dataset = dataflux_pytorch.dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self._args.gcp_project_name,
            bucket_name=self._args.gcs_bucket,
            data_format_fn=format_fn,
            config=dataflux_pytorch.dataflux_mapstyle_dataset.Config(
                prefix=prefix,
                num_processes=self._args.dataflux_num_processes,
                max_composite_object_size=self._args.dataflux_max_composite_object_size,
            )
        )
        if self._args.sample_shuffle != Shuffle.OFF:
            # torch seed is used for all functions within.
            torch.manual_seed(self._args.seed)
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            # generator needs to load up torch seed.
            torch_generator = torch.Generator()
            torch_generator.manual_seed(seed)
            # Pass generator to sampler
            sampler = RandomSampler(df_dataset, generator=torch_generator)
        else:
            sampler = SequentialSampler(df_dataset)
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
            self._dataset = DataLoader(df_dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=True,
                                       drop_last=True, 
                                       **kwargs)
        else: 
            self._dataset = DataLoader(df_dataset,
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
