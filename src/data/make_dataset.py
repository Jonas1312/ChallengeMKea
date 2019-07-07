import os
from shutil import copy2

import pandas as pd
from imagehash import dhash
from PIL import Image


def rearrange_test_set():
    """Merge part_1 and part2 test sets"""

    output_dir = "../../data/interim/TestSetImagesDir/"

    input_dir = "../../data/raw/TestSetImagesDir/part_1/"
    for img_name in os.listdir(input_dir):
        copy2(os.path.join(input_dir, img_name), os.path.join(output_dir))

    input_dir = "../../data/raw/TestSetImagesDir/part_2/"
    for img_name in os.listdir(input_dir):
        copy2(os.path.join(input_dir, img_name), os.path.join(output_dir))

    assert len(os.listdir(output_dir)) == 1715


def rearrange_training_set():
    """To be used with torchvision.datasets.ImageFolder"""

    csv_file = "../../data/raw/TrainingSet_20aimVO.csv"
    df = pd.read_csv(csv_file)
    input_dir = "../../data/interim/TrainingSetImagesDirDuplicatesRemoved/"
    output_dir = "../../data/interim/TrainingSetImagesDirDuplicatesRemovedClasses/"

    for i in range(4):
        output_dir_class = os.path.join(output_dir, str(i))
        if not os.path.exists(output_dir_class):
            os.makedirs(output_dir_class)

    for _, row in df.iterrows():
        img_name = row['image_filename']
        class_number = row['class_number']
        print(img_name, class_number)
        src_path = os.path.join(input_dir, img_name)
        if os.path.exists(src_path):
            copy2(src_path, os.path.join(output_dir, str(class_number)))
        else:
            print("Can't find: ", img_name)


def get_patient_nb(img_name):
    start = img_name.rfind("_")
    return int(img_name[start + 1:-4])


def remove_duplicates():
    """Remove duplicates from training set using dhash"""

    input_dir = "../../data/raw/TrainingSetImagesDir/"
    img_list = os.listdir(input_dir)
    assert len(img_list) == 9446

    output_dir = "../../data/interim/TrainingSetImagesDirDuplicatesRemoved"

    for patient_id in range(56):
        patient_images = [
            x for x in img_list if get_patient_nb(x) == patient_id
        ]
        print(f"Found {len(patient_images)} images for patient {patient_id}")
        if not patient_images:
            continue

        # Compute hash for all images
        hashs = list()
        for img_name in patient_images:
            img_hash = dhash(Image.open(os.path.join(input_dir, img_name)))
            hashs.append(img_hash)

        images_to_delete = set()
        threshold = 5
        for i in range(len(patient_images)):
            for j in range(i + 1, len(patient_images)):
                diff_hash = abs(hashs[i] - hashs[j])
                if diff_hash <= threshold:
                    images_to_delete.add(patient_images[i])
                    break
        print(images_to_delete)

        # Copy unique images
        images_to_copy = set(patient_images) - images_to_delete
        for img_name in images_to_copy:
            copy2(os.path.join(input_dir, img_name), output_dir)


if __name__ == "__main__":
    to_run = (rearrange_training_set, rearrange_test_set, remove_duplicates)
    for func in to_run:
        ret = input(f"Run \"{func.__name__}\"? (y/n) ")
        if "y" in ret:
            func()
            break
