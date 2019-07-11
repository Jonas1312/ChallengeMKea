# Challenge Mauna Kea

[Challenge "Screening and Diagnosis of esophageal cancer" by Mauna Kea](https://challengedata.ens.fr/challenges/11)

## Challenge goals

The goal of this challenge is to build an image classifier to assist physicians in the screening and diagnosis of esophageal cancer. Such a tool would have a massive impact on patient management and patient lives.

<img src="https://diagnosingbarretts.com/images/uploads/Progression-BE.001.png" alt="Illustration" width="600"/>

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

GOAL: 99%
Baseline: 75% (simple CNN)

## Scores updates

| Date  | Model         | LB score | Rank  | Solution                                | weight_name                                                         |
| ----- | ------------- | -------- | ----- | --------------------------------------- | ------------------------------------------------------------------- |
| 06/07 | First commit  | x        | x     |                                         | x                                                                   |
| 11/07 | efficientnet0 | 0.89394  | 21/45 | AdamW<br>MultiStepLR<br>simple data aug | efficientnet_acc=99.13_loss=0.00521_AdamW_ep=17_sz=224_wd=1e-05.pth |
