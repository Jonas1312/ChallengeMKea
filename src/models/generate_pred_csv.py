import os

import pandas as pd
import torch
from torch.nn import functional as F

from architectures.densenet import densenet201
from architectures.efficientnet import efficientnet
from architectures.senet import se_resnet152, se_resnext101_32x4d
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    with_probas = False

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
        dataset = ImageFolderWithPaths(
            "../../data/interim/TestSetImagesDir/",
            transform=transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            ),
            target_transform=None,
        )
        print(f"Found {len(dataset)} images in dataset")

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        model = Model(pretrained=False, num_classes=4).to(device)
        model.load_state_dict(torch.load(os.path.join("../../models/", weights_name)))
        model.eval()

        dic = {}
        processed_images = 0
        with torch.no_grad():
            for data, _, path in test_loader:
                data = data.to(device)
                output = model(data)
                for i, img_path in enumerate(path):
                    img_name = img_path[img_path.find("im_") :]
                    if img_name not in dic:
                        dic[img_name] = [0] * 4
                    if not with_probas:
                        predicted_class = output.max(1, keepdim=True)[1]
                        dic[img_name][predicted_class[i]] += 1
                    else:
                        probas = output[i]
                        probas = F.softmax(probas, dim=-1).cpu().numpy()
                        probas = probas.reshape(-1)
                        for j, prob in enumerate(probas):
                            dic[img_name][j] += prob

                    processed_images += 1
                    print(
                        "Processed {}/{}".format(processed_images, len(dataset)),
                        end="\r",
                    )

    print("")
    print("Get max")
    for key in dic.keys():
        scores = dic[key]
        pred = scores.index(max(scores))
        dic[key] = pred

    # Dict to dataframe
    df = pd.DataFrame.from_dict(dic, orient="index", columns=["class_number"])
    df["image_filename"] = df.index
    df = df[df.columns[::-1]]
    print(df.head())

    # Format dataframe properly
    good_df = pd.read_csv("../../data/raw/test_data_order.csv")
    df = df.set_index("image_filename")
    df = df.reindex(index=good_df["image_filename"])
    df = df.reset_index()
    df.to_csv("../../models/vote_probas_{}.csv".format(with_probas), index=False)


if __name__ == "__main__":
    main()
