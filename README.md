# Human Activity Recognition with Smartphones (Random Forest & SVM)

Phones have sensors inside them. One sensor measures how fast the phone speeds up (the accelerometer) and another sensor measures how the phone turns (the gyroscope). This activity recognition project teaches a computer to look at those sensor numbers and guess what the person was doing. There are six activities: **walking, walking upstairs, walking downstairs, sitting, standing, and laying**.

This repository contains two models built with Python and `scikit-learn`:
1. **Random Forest Classifier** — **92.60% Accuracy**
2. **Support Vector Machine (SVM)** — **95.49% Accuracy**

---

## Project Files & How to Run

* `activity_recognition_random_forest.ipynb`: The notebook containing the Random Forest code.
* `activity_recognition_svm.ipynb`: The notebook containing the SVM code.
* `train.csv`: The data the computer learns from (7,352 rows).
* `test.csv`: The data the project uses to check how well it learned (2,947 rows).
* `README.md`: This project diary and documentation.

### How to Run It
First, open Jupyter. Then upload the notebooks, `train.csv`, and `test.csv` into the folder. If you only have the zip file (`archive.zip`), upload that instead because the notebook can unzip it for you. Then run the cells from top to bottom. If you restart the kernel, you have to run everything from the first cell (I forgot this once and got an error).

---

# Part 1: Activity Recognition using Random Forest (Project Diary)

I used a Random Forest Classifier with Python and `scikit-learn`.

### Entry 1: Finding some data
Every activity recognition project needs data, so that was my first job. I found a file on Kaggle and decided to try it first.

### Entry 2: My first dataset (not the one)
The first file had 88,588 rows and 11 columns. It only had **ONE** person in it and **TWO** activities, labelled `0` and `1`. The file never said what `0` and `1` meant, so I could not be sure which activity was which. That already felt a bit shaky.

Also, each row was one tiny moment and the rows next to each other looked almost the same. That causes a problem called **data leakage**. If I shuffle the rows randomly, the computer can see a row in training that is almost a copy of the one it is being tested on.

So I tried two ways of splitting the data:
* With an **80:20 random split**, I got an accuracy of `0.9898`.
* When I **split by recording day** so the test days were completely new, I got `0.9763`.

The score went down a little, which makes sense because the random split was letting the computer peek. With only one person I could not test on new people, and two activities felt too small. So I switched datasets.

### Entry 3: A better dataset
The new one is called **Human Activity Recognition with Smartphones**. It has 30 volunteers, 6 activities, and 561 features for every row (things like averages and spreads that were already worked out from the sensor signals).

The best part is that the people are split: some are for training, some are for testing, and nobody is in both.

### Entry 4: Looking at