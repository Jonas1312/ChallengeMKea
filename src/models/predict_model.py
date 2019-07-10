"""
  Purpose:  Evaluate on validation set
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from architectures.densenet import densenet169, densenet201
from architectures.senet import se_resnet152, se_resnext101_32x4d
from torchvision import datasets, transforms


def validate(model, device, test_loader, weights):
    model.eval()
    test_loss = 0
    correct = np.zeros(4, dtype=np.int)
    with torch.no_grad():
        for data, target in test_loader:
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

    test_loss /= len(test_loader.dataset)
    nb_samples_per_class = 1 / weights.cpu().numpy()
    nb_samples_per_class = (
        nb_samples_per_class * len(test_loader.dataset) / np.sum(nb_samples_per_class)
    )
    weighted_accuracy = 100 * np.sum(correct / nb_samples_per_class) / len(correct)

    print(
        "Test set: Average loss: {:.6f}, Correct: {}/{}, Weighted accuracy: ({:.2f}%)".format(
            test_loss, np.sum(correct), len(test_loader.dataset), weighted_accuracy
        )
    )
    return test_loss, weighted_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 64
    valid_indices = np.load("../../data/interim/valid_indices.npy")

    weights_files = [
        (densenet169, "densenet169_fold0_acc_99.20_loss_0.005439.pth"),
        (densenet169, "densenet169_fold1_acc_99.41_loss_0.004389.pth"),
        (densenet169, "densenet169_fold2_acc_99.25_loss_0.005233.pth"),
        (se_resnext101_32x4d, "se_resnext101_32x4d_fold4_acc_98.87_loss_0.009070.pth"),
    ]

    for _, weights_name in weights_files:
        if not os.path.isfile(os.path.join("saved_models", weights_name)):
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
