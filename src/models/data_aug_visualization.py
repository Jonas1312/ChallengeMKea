"""
  Purpose:  Test image augmentation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from sampler import ImbalancedDatasetSampler
from torchvision import datasets, transforms


def main():
    batch_size = 4
    input_size = (224,) * 2

    train_set_transforms = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(input_size),
            transforms.RandomAffine(
                degrees=180,
                # translate=(0.05, 0.05),
                # scale=(0.95, 1.05),
                # shear=10,
                resample=2,
                fillcolor=0,
            ),
            # transforms.ColorJitter(
            #     brightness=0.20, contrast=0.15, saturation=0.15, hue=0.04
            # ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.ImageFolder(
        "../../data/interim/TrainingSetImagesDirDuplicatesRemovedClasses/",
        transform=train_set_transforms,
    )
    print("Training set size: ", len(train_set))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=ImbalancedDatasetSampler(train_set),
        batch_size=batch_size,
        num_workers=2,
    )

    count_labels = np.zeros(4, np.int)
    for data, target in train_loader:
        print(data.size(), target.size())
        print(target)
        classes, count = np.unique(target, return_counts=True)
        count_labels[classes] += count
        print(count_labels)
        plt.imshow(data[0, 0, :, :], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
