# Portugal Bank Marketing Campaign

*The dataset for this project originates from the [UCI Portugal bank marketing campaigns Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).*

*Notebook published on Anaconda. [click here](https://anaconda.org/arun2728/portugalbankmarketingcampaigns/notebook)*

#### Task - Predicting whether client will agree to place deposit

## Abstract
In this project, we will evaluate the performance and predictive power of a model that has been trained and tested on data collected from customers of portugal bank. A model trained on this data that is seen as a good fit could then be used to predict if the client will subscribe a term deposit or not.

<hr>

## Dataset

Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature |	Missing Values |
:------------: | :-------------: | :------------: | :-------------: | :------------: | :-------------:
41176 | 21 | 10 | 11 | y | Null

<hr>

## Model Performance

| Model |	AUC Score |	Precision Score	| Recall Score | Accuracy Score	| f1-score |
:------------: | :-------------: | :------------: | :-------------: | :------------: | :-------------:
| KNN |	0.904284	| 0.882394	| 0.931155	| 0.904119	| 0.906119 |

<hr>

## Attribute Information
<br>
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

### Input variables:
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

### Output variable:
 **y** - has the client subscribed a term deposit? (binary: "yes","no")
