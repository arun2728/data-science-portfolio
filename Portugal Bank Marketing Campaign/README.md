# Portugal Bank Marketing Campaign

*The dataset for this project originates from the [UCI Portugal bank marketing campaigns Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).*

*Notebook published on Anaconda. [click here](https://anaconda.org/arun2728/portugalbankmarketingcampaigns/notebook)*

## Background

Portugues Bank conducted a marking campaign to increase the term deposits made by customers. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (‘yes’) or not (‘no’) subscribed.

### Problem Statement:

The bank wants a predictive model which is capable enough to predict whether or not a client will subscribe to the term deposit.



## Dataset

The data is related with direct marketing campaigns of a Portuguese banking institution. 


Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature |
:------------: | :-------------: | :------------: | :-------------: | :------------: | 
41176 | 21 | 10 | 11 | y |



### Attribute Information


- **age** - Age of the customer (numeric)

- **job** - type of job(categorical:"admin.","bluecollar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")

- **marital** - marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)

- **education** - education of individual (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")

- **default** - has credit in default? (categorical: "no","yes","unknown")

- **housing** - has housing loan? (categorical: "no","yes","unknown")

- **loan** - has personal loan? (categorical: "no","yes","unknown")

- **contact** - contact communication type (categorical: "cellular","telephone")

- **month** - last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")

- **dayofweek** - last contact day of the week (categorical: "mon","tue","wed","thu","fri")

- **duration** - last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

- **campaign** - number of contacts performed during this campaign and for this client (numeric, includes last contact)

- **pdays** - number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

- **previous** - number of contacts performed before this campaign and for this client (numeric)

- **poutcome** - outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")

- **emp.var.rate** - employment variation rate - quarterly indicator (numeric)

- **cons.price.idx** - consumer price index - monthly indicator (numeric)

- **cons.conf.idx** - consumer confidence index - monthly indicator (numeric)

- **concave points_se** - standard error for number of concave portions of the contour

- **euribor3m** - euribor 3 month rate - daily indicator (numeric)

- **nr.employed** - number of employees - quarterly indicator (numeric)

#### Target feature:
 **y** - has the client subscribed a term deposit? (binary: "yes","no")

## Exploratory Data Analysis

#### Numeric Features 

- **Age**: It seems that the banks are not very much interested by contacting the older population. Even though, after the 60-years threshold, the relative frequency is higher when y = 1. In other words, we can say that elderly persons are more likely to subscribe to a term deposit.
- **Duration**: It can clearly see that duration attribute highly affects the output target (e.g., if duration = 0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, it should be discarded.
- In **pdays** feature, we should encode '999' as '0' which means that client was not previously contacted as other as '1'
- **pervious** - Even one contact improves probability of “yes” (from 8.8% to 21.2%). But We cannot have a 2nd contact without 1st or a 3rd contact without a 2nd. So we need to perform binning.
- In **emp.var.rate** we should perform logarithmic transformation by taking into consideration the negeative and positives values
- For feature **cons.price.idx**, we should first multiply it by 10 and then perform logarithmic transformation
- In feature **cons.conf.idx** all values are negeative so we should first convert them into positive and then should perform logarithmic transformation
- In feature **nr.employed** the values are on higher scale i.e thousand scale, so they should be reduced on lower scale using logarithmic tranformation
-  Higly correlated features (**employment rate**, **consumer confidence index**, **consumer price index**) may describe clients state from different social-economic angles. Their variance might support model capacity for generalization.


#### Categorical Features

- **job**: Higher response among students (31.4%) and retired people (25.2%). Other classes range between 6.9% (blue-collar) and 14.2 (unemployed).
- **marital**: Singles (14.0%) slightly more like to say “yes” than divorced (10.3%) or married customers (10.2%).
- **default**: Only 3 individuals replied “yes” to the question of having a credit in default. People either answered “no” or didn’t even reply, which gives us zero information. So we can drop this feature.
- **housing**: There is not much observable variation between those who have housing loans (11.6%) and those who do not(10.6%). So we can discard this feature.
- **loan**: There is not much observable variation between those who have personal loans (10.9%) and those who do not(11.3%). So we can discard this feature.
- **contact**: 14.7% of cellular responders subscribed to a term deposit. Only 5.2% of telephone responders did subscribed.
- **month**: 
    - Most of the calls were in May but there is higher percentage of yes from the customer in the month of March, September, October, and in December. 
    - There was no contact made during January and February. The highest spike occurs during May, with 13767 i.e 33.4% of total contacts, but it has the worst ratio of subscribers over persons contacted (6.4%).
    - Every month with a very low frequency of contact (March, September, October and December) shows very good results (between 44% and 51% of subscribers).
- **day_of_week**: Calls aren’t made during weekend days. If we assume that calls are evenly distributed between the different weekdays, Thursdays tend to show better results (12.1% of subscribers among calls made this day) unlike Mondays with only 9.9% of successful calls. However, those differences are small, which makes this feature not that important.

- **poutcome**: 
    - 65.1% of people who already subscribed to a term deposit after a previous contact have accepted to do it again.
    - Even if they were denied before, they’re still more enthusiastic to accept it (14.2%) than people who haven’t been contacted before (8.8%).
    - So even if the previous campaign was a failure, recontacting people seems important.


### Conclusion

In this study four classification algorithms (Logistic Regression, Naive Bayes, Adaboost and KNN) were trained. Precision Recall AUC Score and F1-Score are used as evaluation metrics for our study. Out of all trained classifier KNN outperformed with AUC Score of **0.829841** and F1-score of **0.829841**.

## Model Performance

| Model | AUC Score | Precision Score | Recall Score | Accuracy Score | f1-score |
| :-: | :-: | :-: | :-: | :-: | :-: |
| KNN  | 0.829841 | 0.849699 | 0.800000 | 0.829967 | 0.824101 |

<hr>


