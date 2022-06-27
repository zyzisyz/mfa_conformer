import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.utils.warnings import warn_missing_pkg

from .dataset import Evaluation_Dataset, Train_Dataset, Semi_Dataset


class SPK_datamodule(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        trial_path,
        unlabel_csv_path = None,
        second: int = 2,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        pairs: bool = True,
        aug: bool = False,
        semi: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_csv_path = train_csv_path
        self.unlabel_csv_path = unlabel_csv_path
        self.second = second
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.trial_path = trial_path
        self.pairs = pairs
        self.aug = aug
        print("second is {:.2f}".format(second))

    def train_dataloader(self) -> DataLoader:
        if self.unlabel_csv_path is None:
            train_dataset = Train_Dataset(self.train_csv_path, self.second, self.pairs, self.aug)
        else:
            train_dataset = Semi_Dataset(self.train_csv_path, self.unlabel_csv_path, self.second, self.pairs, self.aug)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        trials = np.loadtxt(self.trial_path, str)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=10,
                                             shuffle=False, 
                                             batch_size=1)
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


