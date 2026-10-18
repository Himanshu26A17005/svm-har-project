# Human Activity Recognition with Smartphones (Random Forest & SVM)

Phones have sensors inside them[cite: 2]. One sensor measures how fast the phone speeds up (the accelerometer) and another sensor measures how the phone turns (the gyroscope)[cite: 2]. The activity recognition project teaches a computer to look at those sensor numbers and guess what the person was doing[cite: 2]. There are six activities: **walking, walking upstairs, walking downstairs, sitting, standing, and laying**[cite: 2].

This repository contains two models built with Python and `scikit-learn`: a **Random Forest Classifier** (92.6% accuracy) and a **Support Vector Machine (SVM)** (95.49% accuracy)[cite: 2].

---

## Project Files & How to Run

* `activity_recognition_random_forest.ipynb`: The notebook with all the Random Forest code[cite: 2].
* `activity_recognition_svm.ipynb`: The notebook with the SVM code.
* `train.csv`: The data the computer learns from (7,352 rows)[cite: 2].
* `test.csv`: The data the project uses to check how well it learned (2,947 rows)[cite: 2].
* `README.md`: This project diary and documentation[cite: 2].

### How to Run It
First, open Jupyter[cite: 2]. Then upload the notebooks, `train.csv`, and `test.csv` into the folder[cite: 2]. If you only have the zip file (`archive.zip`), upload that instead because the notebook can unzip it for you[cite: 2, 4]. Then run the cells from the top to the bottom[cite: 2]. If you restart the kernel, you have to run everything from the first cell (I forgot this once and got an error)[cite: 2].

---

# Part 1: Activity Recognition using Random Forest (Project Diary)

I used something called a Random Forest, with Python and `scikit-learn`[cite: 2].

### Entry 1: Finding some data
Every activity recognition project needs data, so that was my job[cite: 2]. I found a file on Kaggle and decided to try it first[cite: 2].

### Entry 2: My first dataset (not the one)
The first file had 88,588 rows and 11 columns[cite: 2]. It only had **ONE** person in it and **TWO** activities, labelled `0` and `1`[cite: 2]. The file never said what `0` and `1` meant, so I could not be sure which activity was which[cite: 2]. That already felt a bit shaky[cite: 2].

Also, each row was one tiny moment and the rows next to each other looked almost the same[cite: 2]. That causes a problem called **data leakage**[cite: 2]. If I shuffle the rows randomly, the computer can see a row that's almost a copy of the one it is being tested on[cite: 2].

So I tried two ways of splitting the data[cite: 2]:
* With an **80:20 random split**, I got accuracy `0.9898`[cite: 2].
* When I **split by recording day** so the test days were completely new, I got `0.9763`[cite: 2].

The score went down a little, which makes sense because the random split was letting the computer peek[cite: 2]. With only one person I could not test on new people, and two activities felt too small[cite: 2]. So I switched datasets[cite: 2].

### Entry 3: A dataset
The new one is called **Human Activity Recognition with Smartphones**[cite: 2]. It has 30 volunteers, 6 activities, and 561 features for every row (things like averages and spreads that were already worked out from the sensor signals)[cite: 2].

The best part is that the people are split: some are for training, some are for testing, and nobody is in both[cite: 2].

### Entry 4: Looking at the data
`train.csv` has 7,352 rows and 563 columns (561 features, plus `subject`, plus `Activity`)[cite: 2, 4]. `test.csv` has 2,947 rows and 563 columns[cite: 2, 4]. There were no missing values, so I did not have to clean anything[cite: 2, 4]. The six activities were fairly balanced (1,000 to 1,400 rows each in training), so accuracy is a fair score to use[cite: 2, 4].

### Entry 5: Getting X and y ready
* `X` is the 561 feature columns, which is what the computer looks at[cite: 2, 4].
* `y` is the `Activity` column, which is the right answer[cite: 2, 4].

I took the `subject` column out of `X`[cite: 2, 4]. If the computer could see WHO the person is, it might learn the people instead of the movement[cite: 2]. I also took `Activity` out of `X` because that is the answer and it would be cheating![cite: 2, 4]

I did not scale the numbers[cite: 2]. Random Forest does not care how big the numbers are, so scaling is not needed[cite: 2].

### Entry 6: Training the model
A decision tree is like a game of yes/no questions that ends in an