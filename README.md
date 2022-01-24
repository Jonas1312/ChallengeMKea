# Challenge Mauna Kea

[Challenge "Screening and Diagnosis of esophageal cancer" by Mauna Kea](https://challengedata.ens.fr/challenges/11)

## Challenge goals

The goal of this challenge is to build an image classifier to assist physicians in the screening and diagnosis of esophageal cancer. Such a tool would have a massive impact on patient management and patient lives.

<img src="https://live.staticflickr.com/7173/6538264803_121005ed76_b.jpg" alt="Illustration" width="800"/>

## Data description

There are 11161 images acquired from 61 patients to be classified as:

- Class 0: Squamous Epithelium
- Class 1: Intestinal Metaplasia
- Class 2: Gastric Metaplasia
- Class 3: Dysplasia/Cancer

The split between the training and the two test sets is 80%-10%-10%.

The training set is made of 9446 images, acquired from 44 patients:

- Class 0: 1469 images
- Class 1: 3177 images
- Class 2: 1206 images
- Class 3: 3594 images

The two test sets, Test_1 and Test_2, are as follow:

- 893 images acquired from 10 patients
- 822 images acquired from 7 patients

The total is 1715 images acquired from 17 patients.

We also know that if an image acquired from a patient is in the training set, there is no image acquired from the same patient in any test sets.

## Metric

Non weighted Multiclass accuracy = (Number of images correctly classified) / (Number of images in the test set)

Goal: 99%

Baseline: 75% (simple CNN)

## Score updates

| Date       | Model                                                                         | LB score | Rank  | Solution                                     | weight_name                                                         |
| ---------- | ----------------------------------------------------------------------------- | -------- | ----- | -------------------------------------------- | ------------------------------------------------------------------- |
| 06/07      | First commit                                                                  | x        | x     |                                              | x                                                                   |
| 11/07      | EfficientNet0                                                                 | 0.894    | 21/45 | AdamW<br>MultiStepLR<br>simple data aug      | efficientnet_acc=99.13_loss=0.00521_AdamW_ep=17_sz=224_wd=1e-05.pth |
| 13/07      | Ensemble (5):<br>DenseNet<br>SE-ResNext<br>InceptionResNetV2<br>EfficientNet0 | 0.918    | 15/45 | SGD<br>Cosine annealing<br>warm restarts<br> | 5best.csv                                                           |
| 15/07      | Ensemble (3)                                                                  | 0.944    | 8/45  | Pseudo-labeling<br>+ MixUp                   | ch_3best.csv                                                        |
| 16/07      | Ensemble (5)                                                                  | 0.949    | 7/45  | Add 2 models                                 | 2ch_5best.csv                                                       |
| 01/01/2020 | Competition deadline                                                          | x        | x     | x                                            | x                                                                   |

## Final solution

- Data preprocessing:
  - remove duplicates with dhash (9446 images -> 6973 images)
  - split 6973 left images in train/test/valid (80%/10%/10%)
  - No split by patient (I assumed it was not necessary since I had removed nearly identical images, maybe it's a bad idea...)

- Class imbalance:
  - over-sampling during training
  - Weighted CE loss for testing

- Training:
  - simple data aug (flips, random rotations)
  - resized to (224, 224), (299, 299) for InceptionResNetV2
  - Pretrained weights from imagenet
  - SGD
  - LR scheduler : cosine annealing + warm restarts (60 epochs with 3 periods of 20 epochs)
  - no regularization needed

- Pseudo-labeling:
  - on test set (1715 images)
  - predict labels for each image
  - if all (or more than X%) images that belong to the same patient have the same prectited class => add patient to train set

- Ensemble:
  - with probabilites from softmax

## What didn't work

- Focal loss
- Adam/AdamW (converges fast but not as good as SGD)
- Adabound/Amsbound

## Things I would have liked to try if GPU cloud providers weren't so expensive 😕

- Other EfficientNet models
- TTA
- For pseudo-labeling:
  - https://arxiv.org/abs/1904.12848v2
  - https://arxiv.org/abs/1905.02249v1
- Hard mining
- ridge regression : http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/
