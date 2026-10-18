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

### Entry 4: Looking at the data
`train.csv` has 7,352 rows and 563 columns (561 features, plus `subject`, plus `Activity`). `test.csv` has 2,947 rows and 563 columns. There were no missing values, so I did not have to clean anything. The six activities were fairly balanced (1,000 to 1,400 rows each in training), so accuracy is a fair score to use.

### Entry 5: Getting X and y ready
* `X` is the 561 feature columns, which is what the computer looks at.
* `y` is the `Activity` column, which is the right answer.

I took the `subject` column out of `X`. If the computer could see WHO the person is, it might learn the people instead of the movement. I also took `Activity` out of `X` because that is the answer and it would be cheating!

I did not scale the numbers. Random Forest does not care how big the numbers are, so scaling is not needed.

### Entry 6: Training the model
A decision tree is like a game of yes/no questions that ends in an answer. A Random Forest grows 100 of these trees, and then they all vote. The answer with the most votes wins. Lots of trees voting makes fewer mistakes than one tree on its own.

In code, there are two lines:
* `RandomForestClassifier(n_estimators=100, random_state=42)` builds the forest.
* `model.fit(X_train, y_train)` trains the model. The word "fit" means the computer looks at the training rows and their correct answers and learns the patterns.

### Entry 7: The test!
I gave the model the numbers from `test.csv` without any answers and asked the model to guess. The test people were never seen during training, making this a fair test. These are the results from running the code:

* **Accuracy:** `0.926` (92.6% correct)
* **Precision (macro):** `0.9274`
* **Recall (macro):** `0.9232`
* **F1-score (macro):** `0.9245`

**What these numbers mean in words:**
* **Accuracy** tells how many answers were correct out of all answers.
* **Precision** tells how often the model is correct when it predicts a specific activity (like "walking").
* **Recall** tells how many of the actual walking rows were found by the model.
* **F1-score** is one number that balances precision and recall.
* **Macro** means the model calculated each metric for every activity and then took the average.

### Entry 8: Where did it get confused?
My confusion matrix is a 6 by 6 table. The rows show the true activity and the columns show what the model guessed. The correct guesses appear on the diagonal.

```text
Labels order: ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

[[537   0   0   0   0   0]
 [  0 435  56   0   0   0]
 [  0  41 491   0   0   0]
 [  0   0   0 482   6   8]
 [  0   0   0  15 365  40]
 [  0   0   0  46   6 419]]