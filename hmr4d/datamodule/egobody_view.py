import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import DictConfig
from numpy.random import choice
from hmr4d.utils.pylogger import Log
from hmr4d.datamodule.mocap_trainX_testY import collate_fn


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_opts: DictConfig, loader_opts: DictConfig, limit_each_trainset=None):
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset

        # Train
        if "train" in dataset_opts:
            split_opts = dataset_opts.get("train")
            dataset = []
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                if self.limit_each_trainset:
                    dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                dataset.append(dataset_i)
                Log.info(f"[Train Dataset][{idx+1}/{len(split_opts)}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            self.trainset = ConcatDataset(dataset)

        # Val
        if "val" in dataset_opts:
            split_opts = dataset_opts.get("val")
            dataset = []
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                dataset.append(dataset_i)
                Log.info(f"[Val Dataset][{idx+1}/{len(split_opts)}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            self.valsets = dataset

        # Test
        if "test" in dataset_opts:
            split_opts = dataset_opts.get("test")
            dataset = []
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                dataset.append(dataset_i)
                Log.info(f"[Test Dataset][{idx+1}/{len(split_opts)}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            self.testsets = dataset

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            shuffle=True,
            num_workers=self.loader_opts.train.num_workers,
            persistent_workers=True and self.loader_opts.train.num_workers > 0,
            batch_size=self.loader_opts.train.batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        loaders = []
        for valset in self.valsets:
            loaders.append(
                DataLoader(
                    valset,
                    shuffle=False,
                    num_workers=self.loader_opts.val.num_workers,
                    persistent_workers=True and self.loader_opts.val.num_workers > 0,
                    batch_size=self.loader_opts.val.batch_size,
                    collate_fn=collate_fn,
                )
            )
        return CombinedLoader(loaders, mode="sequential")

    def test_dataloader(self):
        loaders = []
        for testset in self.testsets:
            loaders.append(
                DataLoader(
                    testset,
                    shuffle=False,
                    num_workers=self.loader_opts.test.num_workers,
                    persistent_workers=True and self.loader_opts.test.num_workers > 0,
                    batch_size=self.loader_opts.test.batch_size,
                    collate_fn=collate_fn,
                )
            )
        return CombinedLoader(loaders, mode="sequential")
