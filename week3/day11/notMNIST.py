import os
import numpy as np

from typing import Callable, Optional, Dict, Tuple, Any

from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class NotMNIST(VisionDataset):

    resources = [
        ("http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz"),
        ("http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. \nYou can use download=True to download it."
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: idx for idx, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def read_data(self, path: str):
        images = []
        labels = []

        for i, c in enumerate(self.classes):
            for imagename in os.listdir(os.path.join(path, c)):
                with open(os.path.join(path, c, imagename), "rb") as f:
                    try:
                        image = Image.open(f)
                        images.append(np.asarray(image.convert("RGB")))
                        labels.append(i)
                    except:
                        print(f"{os.path.join(path, c, imagename)} file maybe empty.")
        return (
            torch.from_numpy(np.asarray(images)),
            torch.from_numpy(np.asarray(labels)),
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition("/")[-1]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename
            )

        # process and save as torch files
        print("Processing...")

        training_set = self.read_data(os.path.join(self.raw_folder, "notMNIST_large"))
        test_set = self.read_data(os.path.join(self.raw_folder, "notMNIST_small"))
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")
