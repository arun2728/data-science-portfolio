# Personal Loan Acceptance

*The dataset for this project originates from the [Universal bank data for classification](https://www.kaggle.com/sriharipramod/bank-loan-classification).*

*Notebook published on Anaconda. [click here](https://anaconda.org/arun2728/loanacceptance)*

## Background

This case is about a bank (Universal bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns to better target marketing to increase the success ratio with a minimal budget.


### Problem Statement:
The department wants to build a model that will help them identify the potential customers who have a higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.

## Dataset
The file **UniversalBank.csv** contains data on **5000** customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan).

| Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature |
| :-: | :-: | :-: | :-: | :-: |
| 5000  | 12 | 5 | 7  | PersonalLoan |

### Attribute Information

1) **ID** (Customer ID)


2) **Age:** Customer's age in completed years


3) **Experience:** years of professional experience


4) **Income:** Annual income of the customer ($000)


5) **ZIP Code:** Home Address ZIP code.


6) **CCAvg:** Avg. spending on credit cards per month ($000)


7) **Education:** Education Level - 1 : Undergrad, 2 : Graduate, 3 : Professional


8) **Mortgage:** Value of house mortgage if any. ($000)


9) **Family:** Family size of the customer


10) **Securities Account:** Does the customer have a securities account with the bank?


11) **CD Account:** Does the customer have a certificate of deposit (CD) account with the bank?


12) **Online:** Does the customer use internet banking facilities?


13) **Credit card:** Does the customer use a credit card issued by


#### Target variable (desired target):

14) **Personal Loan:** Did this customer accept the personal loan offered in the last campaign?


## Exploratory Data Analysis


#### Numeric Features 

- **Age** feature is normally distributed with majority of customers falling between **30 years and 60 years** of age. We can confirm this by looking at the describe statement above, which shows mean is almost equal to median

- **Experience** is normally distributed with more customer having experience starting from **8 years**. Here the mean is equal to median. There are negative values in the Experience. 

    - This could be a data input error as in general it is not possible to measure negative years of experience. We will replace with the negeative records with median of entries having save age as age and experience are related to each other.

- Additionally, scatter plot of **Age** and **experience** indicated that they are positively correlated. As experience increase age also increases. 

- **Income** is positively skewed. Majority of the customers have income between **45K** and **55K**. We can confirm this by saying the mean is greater than the median

- **CCAvg** is also a positively skewed variable and average spending is between **0K to 10K** and majority spends less than **2.5K**

- Customers having a personal loan have a higher credit card average. Average credit card spending with a median of **3800** dollar indicates a **higher probability** of personal loan. Lower credit card spending with a median of **1400** dollars is less likely to take a loan. 

- **70%** of the individuals have a **mortgage** of less than **40K**. However the max value is **635K**.

- **Income** and **CCAvg** are moderately correlated
- **Age** and **Experience** are highly correlated


#### Categorical Features

- Majority of customers who does not have loan have securities account

- Family size does not have any impact in personal loan. But it seems families with size of 3 are more likely to take loan. When considering future campaign this might be good association.

- Customers who does not have CD account , does not have loan as well. This seems to be majority. But almost all customers who has CD account has loan as well.

- The customers whose education level is 1 is having more income. However customers who has taken the personal loan have the same income levels.

- Additionally, Customers having personal loan have high mortgage

- Generally, as family size increase we can see that the education qualification decrease. Family with more number of members don't opte for higher studies.

- Generally, Larger familes prefer to have a certificate of deposit (CD) account with the bank. And majority of customers not having a Securities Account don't have CD Account

- Customers using a credit card don't opt for a certificate of deposit account with the bank also the one using a internet banking facilities have a certificate of deposit (CD) account with the bank.


## Conclusion

The aim of the universal bank is to convert there liability customers into loan customers. They want to set up a new marketing campaign; hence, they need information about the connection between the variables given in the data. 

In this study four classification algorithms (Logistic Regression, Naive Bayes, Support Vector Machine and Random Forest) were trained. Precision Recall AUC Score and F1-Score are used as evaluation metrics for our study. Out of all trained classifier **Random Forest** outperformed with **PR AUC Score** of **0.95** and **F1-score** of **0.95**.

According to random forest classifier, target feature heavily depends on

- **Income** - Annual income of the customer
- **CCAvg** - Avg. spending on credit cards per month 
- **Education** - Education Level of the customer {1 : Undergrad, 2 : Graduate, 3 : Advanced/Professional}
- **Family** - Family size of the customer
- **Age** - Customer's age in completed years

#### Trained Models -

| Models	| PR-AUC Score	| Precision	| Recall 	| f1-score |
| :- | :- | :- | :- | :- |
| Random Forest	| 0.9556	| 0.9702	| 0.9333	| 0.9514 |
| Logistic Regression	| 0.7828	| 0.8554	| 0.6761	| 0.7553 |
| Naive Bayes	| 0.5733	| 0.4961	|  0.6095	 |0.5479 |
| Support Vector Machine | 0.9032	| 0.9771 | 0.8095 | 0.885 |

#### Precision Recall AUC Plot -

![no image](https://github.com/arun2728/data-science-portfolio/blob/main/Loan%20Acceptance/model/models_roc_plot.png)
