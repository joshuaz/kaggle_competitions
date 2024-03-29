## The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Dataset Description
### Overview
The data has been split into two groups:

- training set (train.csv)
- test set (test.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

### Data Dictionary
| Variable	   | Definition| Key        |
|--------------|-----------|------------|
| survival	   | Survival	 | 0 = No, 1 = Yes|
| pclass       | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd|
| sex          | Sex       |            |
| Age          | Age in years|            |
| sibsp	       | # of siblings / spouses aboard the Titanic |            |
| parch	       | # of parents / children aboard the Titanic |            |
| ticket       | Ticket number |            |
| fare         | Passenger fare |            |
| cabin	       | Cabin number   |            |
| embarked     | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
