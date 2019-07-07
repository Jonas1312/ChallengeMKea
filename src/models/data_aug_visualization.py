# coding:utf-8
"""
  Purpose:  Test image augmentation
"""

from collections import Counter

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from sampler import ImbalancedDatasetSampler


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
        num_workers=4,
    )

    nb_0, nb_1, nb_2, nb_3 = (0,) * 4
    for data, target in train_loader:
        print(data.size(), target.size())
        print(target)
        z = target.tolist()
        z = Counter(z)
        nb_0 += z[0]
        nb_1 += z[1]
        nb_2 += z[2]
        nb_3 += z[3]
        print(nb_0, nb_1, nb_2, nb_3)
        plt.imshow(data[0, 0, :, :], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
