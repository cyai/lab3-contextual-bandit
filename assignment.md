# Lab 3: Contextual Bandit-Based News Article Recommendation

# System

## Reinforcement Learning Fundamentals

```
Total: 105 Points February 8, 2026
```

# 1 Introduction

Contextual Multi-Armed Bandits (CMAB) extend the classic Multi-Armed Bandit (MAB) framework
by introducing “side information” orcontextsto the decision-making process. While a standard
MAB agent operates in a stateless environment where rewards depend solely on the chosen action
(Arm), a CMAB agent observes a specificcontext(e.g., user demographics or time of day) before
choosing an action. In this setting, the goal shifts from finding the single best arm overall to learning
a policy that selects the optimal arm conditioned on the current context, effectively mapping specific
situations to the actions that yield the highest expected reward.

# 2 Objective

The primary objective of this assignment is to design and implement aNews Recommendation
Systemutilizing aContextual Banditframework. The system will train a Reinforcement Learning
(RL) model to recommend news articles by treating user categories ascontextsand news categories
as arms.
Given a new user, the system must predict the most suitable news category and sample a specific
article to maximize user engagement (reward).

# 3 Environment & Dataset Specifications

## 3.1 Datasets

```
You are provided with two primary datasets:
```

- News Articles Dataset(news_articles.csv): Each row contains anews articlewith
  variousfeaturesand includes a label columncategoryspecifying the news category (represents
  theArmsof the bandit).
- User Data(train_users.csv,test_users.csv): Each row represents an individualuser
  with variousfeatures, including alabelcolumn classifying the user intoUser1,User2, or
  User3(representing theContexts).

## 3.2 Reward Distribution Guide (Sampler Utility)

```
To simulate the environment’s response (rewards), you are provided with a Python package named
rlcmab-sampler. This package contains a Python classsamplerused to fetch rewards from unknown
probability distributions. Students must use the provided package as-is. Any modification or
reimplementation of the sampler will result in zero credit for the bandit component.
```

Installation
UsePython>=3.12to install the package usingpip:

```
pip install rlcmab -sampler
```

Usage
Thesampleris initialized with your student Roll Number (i) and queried using an arm index (j).

```
from rlcmab_sampler import sampler
# Initialize with your roll number (i)
reward_sampler = sampler(i)
# Call the function to get a reward from arm j
reward = reward_sampler.sample(j)
```

```
Student ID Mapping
ID Number Roll Number (i)
U202300078 78
U202300115 115
U202300001 1
```

# 4 Problem Formulation

The environment is modeled as aContextual Banditproblem with the following structure:

- Contexts:3 unique user types (User1,User2,User3).
- Bandits: 4 distinct news categories per context.
- Total Arms: 3 contexts× 4 categories= 12arms.
  The arm indexjpassed to the ‘sample(j)‘ function must map to the specific combination of User
  Context and News Category as defined below:

```
jValues Configuration (News Category, User Context)
{ 0 , 1 , 2 , 3 } {(Entertainment, User1), (Education, User1), (Tech, User1), (Crime, User1)}
{ 4 , 5 , 6 , 7 } {(Entertainment, User2), (Education, User2), (Tech, User2), (Crime, User2)}
{ 8 , 9 , 10 , 11 } {(Entertainment, User3), (Education, User3), (Tech, User3), (Crime, User3)}
Table 1: Arm Index (j) Mapping
```

# 5 Implementation Tasks

## 5.1 Data Pre-processing (10 Points)

1. Load the provided user and article datasets.
2. Perform necessary data cleaning (e.g., handling missing values).
3. Applyfeature encodingwhere required to prepare the data for classification and bandit
   training.

## 5.2 User Classification (10 Points)

Develop a classification model (e.g., Decision Tree, Logistic Regression) to predict the user category
(User1,User2, orUser3) based on input feature data.

- Split thetrain_users.csvdataset into a training set (80%) and a validation set (20%) for
  model evaluation.
- The model must be trained on the training set and evaluated on the validation set to ensure it
  can accurately classify users into their respective categories.
- This classifier will serve as the “Context Detector” for your bandit system where you will use
  test_users.csvdataset.

## 5.3 Contextual Bandit Algorithms (45 Points)

You must implement three distinct strategies. For each strategy, treat theUser Categoryas the
contextandNews Categoryas theArm.

```
5.3.1 Epsilon-Greedy (15 Points)
```

- Train a separate model for each of the 3 user contexts.
- Compute theExpected Reward Distributionfor each news category across all contexts.
- Hyperparameter Tuning:Experiment with multiple values ofε. Compare the expected
  payoffs for differentεvalues.
  5.3.2 Upper Confidence Bound (UCB) (15 Points)
- Train a separate model for each of the 3 user contexts.
- Compute theExpected Reward Distribution.
- Hyperparameter Tuning: Experiment with multiple values of the exploration parameterC.
  Compare expected payoffs.
  5.3.3 SoftMax (15 Points)
- Train a separate model for each of the 3 user contexts.
- Use a fixed temperature parameterτ= 1.
- Compute theExpected Reward Distribution.

## 5.4 Recommendation Engine (20 Points)

Consolidate the classification and decision-making components to establish the end-to-end operational
workflow for the CMAB recommendation engine:

1. Classify:Determine the User Category using the model from 5.2.
2. Select Category: Use the trained Bandit Policies from 5.3 to select the optimalNews
   Category.
3. Recommend:Randomly sample an article from the selected category innews_articles.csv.
4. Input/Output:System inputs user features fromtest_users.csvand output the optimal
   news category plus a sampled article..

## 5.5 Evaluation & Reporting (20 Points)

1. Classification Accuracy:Evaluate the classifier on a20%validation split oftrain_users.csv
   usingsklearn.metrics.classification_report.
2. RL Simulation:Run the RL models for a time horizon ofT= 10, 000 steps.
3. Analysis Plots:
    - PlotAverage Reward vs. Timefor each context.
    - PlotAverage Reward comparisonfor different hyperparameters (εandC). Test at
      least 3 distinct values for each.
4. Final Report: Compile a comprehensive analysis of the three models, observations on
   hyperparameter sensitivity, and comparative performance.

# 6 Submission Guidelines

- Repository:Students mustforkthis repository into their own GitHub account.
- Branching: All work must be completed in the forked repository and pushed to a branch
  namedfirstname_U20230xxx. Do not push to the masterbranch. Submissions on
  masterwill be ignored.
- Contents:
    - An IPython Notebook (.ipynb) file placed at theroot of the repository, named
      lab3*results*<roll_number>.ipynb. AREADME.mdserving as the project report.
    - The notebook should contain all code, results, and visualizations. The README should
      summarize the approach, results, and insights.
    - Each plot must include labeled axes, a legend, and a descriptive title. Unlabeled or
      unclear plots may receive partial or no credit.
- Final Requirements:Before submission, ensure that your code runs without errors and that
  all plots are correctly generated. Failure to meet these requirements may result in a significant
  deduction of points.
