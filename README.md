# Naive Bayes Classification with Scikit-learn

A hands-on classification project exploring Bayes' theorem and Naive Bayes models with Python and Scikit-learn. The notebook applies Gaussian Naive Bayes to a small weather example and the Wine dataset, then uses a text-appropriate Naive Bayes model to classify SMS messages as ham or spam.

## What this project covers

- Encoding categorical values and preparing feature and target data.
- Training and evaluating a Gaussian Naive Bayes classifier on binary and multiclass classification problems.
- Splitting data into training and test sets and measuring accuracy.
- Converting SMS text into bag-of-words features with Scikit-learn's `CountVectorizer`.
- Comparing the modeling assumptions behind Gaussian and Multinomial Naive Bayes.

## Models and results

| Dataset | Features | Model | Evaluation |
|---|---|---|---|
| Weather example | Encoded weather and temperature | `GaussianNB` | Example predictions for the `play` label |
| Wine | 13 numeric chemical measurements | `GaussianNB` | About 90.7% accuracy with a 30% test split and `random_state=109` |
| SMS Spam Collection | Bag-of-words message counts | `MultinomialNB` | About 98.9% accuracy with `random_state=1` |

The SMS section uses `MultinomialNB`, rather than `GaussianNB`, because the message features are word counts. Accuracy is dataset- and split-dependent; the figures above correspond to the notebook's specified splits.

## Repository contents

- `Module20_assignment_student.ipynb` — completed Jupyter notebook with code and written responses.
- `SMSSpamCollection` — SMS dataset used in the final exercise.

## Getting started

Python 3 and Jupyter are required. From the project directory, install the dependencies and start Jupyter:

```bash
python -m pip install numpy pandas scikit-learn jupyter
jupyter notebook
```

Open `Module20_assignment_student.ipynb` and run the cells from top to bottom. Keep `SMSSpamCollection` in the same directory as the notebook so the data-loading cell can find it.

## Data source and acknowledgements

The SMS Spam Collection is associated with the UCI Machine Learning Repository. See the [dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) for dataset details and usage information. Retain appropriate attribution and review the dataset's terms before redistributing it.

This notebook was completed as a Module 20 assignment on machine learning and advanced probability.
