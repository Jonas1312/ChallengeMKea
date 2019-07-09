# coding: utf-8
"""
Purpose: Train model
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from architectures.efficientnet import efficientnet as Model
from sampler import ImbalancedDatasetSampler
from torchvision import datasets, transforms


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    nb_samples = 0
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = F.cross_entropy(output, target, reduction="sum")
        epoch_loss += loss.item()
        loss /= len(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                epoch,
                nb_samples,
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                loss.item(),
            ),
            end="\r",
        )

    epoch_loss /= len(train_loader.dataset)
    print(
        "Train Epoch: {} [{}/{} ({:.0f}%)], Average Loss: {:.6f}".format(
            epoch, nb_samples, len(train_loader.dataset), 100.0, epoch_loss
        )
    )
    return epoch_loss


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
            pred = output.max(dim=1, keepdim=True)[1]
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
    print("Accuracy per class: ", 100 * correct / nb_samples_per_class)
    print(
        "Test set: Average loss: {:.6f}, Correct: {}/{}, Weighted accuracy: ({:.2f}%)".format(
            test_loss, np.sum(correct), len(test_loader.dataset), weighted_accuracy
        )
    )
    return test_loss, weighted_accuracy


def checkpoint(model, test_loss, test_acc, optimizer, epoch, input_size, weight_decay):
    file_name = "{}_acc:{:.2f}_loss:{:.6f}_{}_ep:{}_sz:{}_wd{}:.pth".format(
        Model.__name__,
        test_acc,
        test_loss,
        optimizer.__class__.__name__,
        epoch,
        input_size[0],
        weight_decay,
    )
    path = os.path.join("../../models/", file_name)
    if test_acc > 95 and not os.path.isfile(path):
        torch.save(model.state_dict(), path)
        print("Saved: ", file_name)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyperparams
    batch_size = 32
    epochs = 60
    input_size = (224,) * 2
    weight_decay = 1e-5
    print(f"Batch size: {batch_size}, input size: {input_size}")

    # Create datasets
    train_indices = np.load("../../data/interim/train_indices.npy")
    test_indices = np.load("../../data/interim/test_indices.npy")
    assert not set(train_indices) & set(test_indices)

    # Transforms
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
    test_set_transforms = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize(input_size), transforms.ToTensor()]
    )

    # Datasets
    train_set = torch.utils.data.Subset(
        datasets.ImageFolder(
            "../../data/interim/TrainingSetImagesDirDuplicatesRemovedClasses/",
            train_set_transforms,
        ),
        train_indices,
    )
    test_set = torch.utils.data.Subset(
        datasets.ImageFolder(
            "../../data/interim/TrainingSetImagesDirDuplicatesRemovedClasses/",
            test_set_transforms,
        ),
        test_indices,
    )
    print("Training set size: ", len(train_set))
    print("Test set size : ", len(test_set))
    print("Total: ", len(train_set) + len(test_set))

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,  # shuffling is handled by sampler
        sampler=ImbalancedDatasetSampler(train_set, train_indices),
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Distribution of classes in the test set
    label_counter = {0: 0, 1: 0, 2: 0, 3: 0}
    for indice in test_set.indices:
        label = test_set.dataset.imgs[indice][1]
        label_counter[label] += 1
    weights = dict(sorted(label_counter.items())).values()  # sort by keys
    weights = torch.FloatTensor([1.0 / x for x in weights]).to(device)
    weights = weights / torch.sum(weights)
    print("Class weights: ", weights)

    model = Model(pretrained=True, num_classes=4).to(device)
    print(Model.__name__)

    # optimizer = torch.optim.SGD(
    # model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=weight_decay
    )
    print("Optimizer: ", optimizer.__class__.__name__)
    # scheduler = CosineAnnealingRestartsLR(
    #     optimizer, T=20, eta_min=0, T_mult=1.0, eta_mult=0.3
    # )

    # Train first and last layer for one epoch!!!

    train_loss_history = list()
    test_loss_history = list()
    accuracy_history = list()

    for epoch in range(1, epochs + 1):
        print("################## EPOCH {}/{} ##################".format(epoch, epochs))

        # scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group["lr"])

        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, acc = validate(model, device, test_loader, weights)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        accuracy_history.append(acc)

        # Save history at each epoch (overwrite previous history)
        history = [train_loss_history, test_loss_history, accuracy_history]
        np.save("history.npy", np.array(history))


if __name__ == "__main__":
    main()
