import os
from shutil import copy2

import pandas as pd


def rearrange_training_set():
    """To be used with torchvision.datasets.ImageFolder"""

    csv_file = "../../data/raw/TrainingSet_20aimVO.csv"
    df = pd.read_csv(csv_file)
    input_dir = "../../data/raw/TrainingSetImagesDir/"
    output_dir = "../../data/interim/TrainingSetImagesDirClasses/"

    for i in range(4):
        output_dir_class = os.path.join(output_dir, str(i))
        if not os.path.exists(output_dir_class):
            os.makedirs(output_dir_class)

    for _, row in df.iterrows():
        img_name = row['image_filename']
        class_number = row['class_number']
        print(img_name, class_number)
        copy2(os.path.join(input_dir, img_name),
              os.path.join(output_dir, str(class_number)))


if __name__ == "__main__":
    to_run = (rearrange_training_set, )
    for func in to_run:
        ret = input(f"Run \"{func.__name__}\"? (y/n)")
        if "y" in ret:
            func()
