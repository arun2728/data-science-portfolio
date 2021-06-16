# Credit Card Customers Segmentation

*The dataset for this project originates from the [Credit Card Dataset for Clustering
](https://www.kaggle.com/arjunbhasin2013/ccdata).* 
<br>


### Background:

Not all customers are alike. Consumers usually show a wide variety of behaviors. A lot of times, Segments that are used in businesses are threshold based. With growing number of features and a general theme of personlized products, there is a need for a scietific based methodology to group customers together. Clustering based on the behavioral data comes to the rescue. The aim of this analysis is to group credit card holders in appropriate groups to better understand their needs and behaviors and to serve them better with appropriate marketing offers.

### Problem Statement: 
In this project, we need to extract segments of customers depending on their behaviour patterns provided in the dataset, to focus marketing strategy of the company on a particular segment.

<hr>

## Dataset

Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature |	Missing Values |
:------------: | :-------------: | :------------: | :-------------: | :------------: | :-------------:
41176 | 21 | 10 | 11 | y | Null

<hr>

### Attribute Information:


1) *CUSTID:* Identification of Credit Card holder (Categorical)


2) *BALANCE:* Balance amount left in their account to make purchases


3) *BALANCE_FREQUENCY:* How frequently the Balance is updated, score between 0 and 1

4) *PURCHASES:* Amount of purchases made from account

5) *ONEOFF_PURCHASES:* Maximum purchase amount done in one-go

6) *INSTALLMENTS_PURCHASES:* Amount of purchase done in installment

7) *CASH_ADVANCE:* Cash in advance given by the user

8) *PURCHASES_FREQUENCY:* How frequently the Purchases are being made, score between 0 and 1 

9) *ONEOFF_PURCHASES_FREQUENCY:* How frequently Purchases are happening in one-go 

10) *PURCHASES_INSTALLMENTS_FREQUENCY:* How frequently purchases in installments are being done 

11) *CASH_ADVANCE_FREQUENCY:*  How frequently the cash in advance being paid

12) *CASH_ADVANCE_TRX:* Number of Transactions made with "Cash in Advanced"

13) *PURCHASES_TRX:* Number of purchase transactions made

14) *CREDIT_LIMIT:* Limit of Credit Card for user

15) *PAYMENTS:* Amount of Payment done by user

16) *MINIMUM_PAYMENTS:* Minimum amount of payments made by user

17) *PRC_FULL_PAYMENT:* Percent of full payment paid by user

18) *TENURE:* Tenure of credit card service for user


<hr>

## Clusters

![alt text](https://github.com/arun2728/data-science-portfolio/blob/main/Customer%20Segmentation/output/cluster.png)
<hr>

## Conclusion


#### Large segments:

- **Cluster 1**: This group of customers on the other hand are not completely utilizing the credit line assigned to them. Additional investigations are needed to understand why this particular set of consumers are not utilizing their lines or if their credit lines could in the future be assigned to a different set of consumers.

- **Cluster 2**: This group of customers is in a dire need of a credit limit increase. They also have the highest activities among all the clusters.

- **Cluster 0**: This cluster belongs to customers with adequate activites and balance.

- **Cluster 5**: This cluster shows slightly higher balances and purchase activities, but higher one-off purchase behavior.

#### Small segments:

- **Cluster 3**: This cluster shows low balances but average activity. This cluster will be an approprite cluster for spend campaign targeting.

- **Cluster 4**: This cluster has the highest activity, balances, and purchases. This group of customers interestingly also have a higher set of credit lines, indicating that an increasing credit limit increases leads to an increase in the purchase activities. (A rigourous testing of this hypothesis should be carries out.)




