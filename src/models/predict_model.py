"""
  Purpose:  Evaluate on validation set
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from architectures.efficientnet import efficientnet
from torchvision import datasets, transforms


def validate(model, device, test_loader, weights):
    model.eval()
    nb_samples = 0
    test_loss = 0
    correct = np.zeros(4, dtype=np.int)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            nb_samples += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, weight=weights, reduction="sum"
            ).item()
            pred = output.max(1, keepdim=True)[1]
            pred = pred.view_as(target)
            good_preds = target[pred == target].cpu().numpy()
            classes, count = np.unique(good_preds, return_counts=True)
            correct[classes] += count

            print(
                "[{}/{} ({:.0f}%)]".format(
                    nb_samples,
                    len(test_loader.dataset),
                    100.0 * (batch_idx + 1) / len(test_loader),
                ),
                end="\r",
            )

    test_loss /= len(test_loader.dataset)
    nb_samples_per_class = 1 / weights.cpu().numpy()
    nb_samples_per_class = (
        nb_samples_per_class * len(test_loader.dataset) / np.sum(nb_samples_per_class)
    )
    weighted_accuracy = 100 * np.sum(correct / nb_samples_per_class) / len(correct)

    print(
        "\nTest set: Average loss: {:.6f}, Correct: {}/{}, Weighted accuracy: ({:.2f}%)".format(
            test_loss, np.sum(correct), len(test_loader.dataset), weighted_accuracy
        )
    )
    return test_loss, weighted_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 128
    valid_indices = np.load("../../data/interim/valid_indices.npy")

    weights_files = [
        (
            efficientnet,
            "efficientnet_acc=99.13_loss=0.00521_AdamW_ep=17_sz=224_wd=1e-05.pth",
        )
    ]

    for _, weights_name in weights_files:
        if not os.path.isfile(os.path.join("../../models/", weights_name)):
            raise FileNotFoundError(weights_name)

    for Model, weights_name in weights_files:
        print(weights_name)
        input_size = (224,) * 2
        if "inceptionresnetv2" in weights_name:
            input_size = (299,) * 2

        valid_set_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )

        valid_set = torch.utils.data.Subset(
            datasets.ImageFolder(
                "../../data/interim/TrainingSetImagesDirDuplicatesRemovedClasses/",
                valid_set_transforms,
            ),
            valid_indices,
        )

        print("Valid set size: ", len(valid_set))

        test_loader = torch.utils.data.DataLoader(
            dataset=valid_set, batch_size=batch_size, num_workers=4, pin_memory=True
        )

        # Distribution of classes in the test set
        label_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        for indice in valid_set.indices:
            label = valid_set.dataset.imgs[indice][1]
            label_counter[label] += 1
        weights = dict(sorted(label_counter.items())).values()  # sort by keys
        weights = torch.FloatTensor([1.0 / x for x in weights]).to(device)
        weights = weights / torch.sum(weights)
        print("Class weights: ", weights)

        model = Model(pretrained=False, num_classes=4).to(device)
        model.load_state_dict(torch.load(os.path.join("../../models/", weights_name)))

        validate(model, device, test_loader, weights)


if __name__ == "__main__":
    main()
