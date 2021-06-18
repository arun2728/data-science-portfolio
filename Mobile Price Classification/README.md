# Portugal Bank Marketing Campaign

*The dataset for this project originates from the [Classify Mobile Price Range](https://www.kaggle.com/iabhishekofficial/mobile-price-classification).*

*Notebook published on Anaconda. [click here](https://anaconda.org/arun2728/mobilepriceclassification/notebook)*

## Context
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc. He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

### Problem Statement:
In this problem you do not have to predict actual price but a price range indicating how high the price is

## Dataset

Number of Instances | Number of Attributes | Numeric Features | Categorical Features | Target Feature |	Missing Values |
:------------: | :-------------: | :------------: | :-------------: | :------------: | :-------------:
2000 | 21 | 13 | 8 | price_range | Null


## Attribute Information
<br>
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

### Input variables:
- **battery_power** - Total energy a battery can store in one time measured in mAh

- **blue** - Has bluetooth or not

- **clock_speed** - speed at which microprocessor executes instructions

- **dual_sim** - Has dual sim support or not

- **fc** - Front Camera mega pixels

- **four_g** - Has 4G or not

- **int_memory** - Internal Memory in Gigabytes

- **m_dep** - Mobile Depth in cm

- **mobile_wt** - Weight of mobile phone

- **n_cores** - Number of cores of processor

- **pc** - Primary Camera mega pixels
- **px_height** - Pixel Resolution Height
- **px_width** - Pixel Resolution Width
- **ram** - Random Access Memory in Mega Bytes
- **sc_h** - Screen Height of mobile in cm
- **sc_w** - Screen Width of mobile in cm
- **talk_time** - longest time that a single battery charge will last when you are
- **three_g** - Has 3G or not
- **touch_screen** - Has touch screen or not
- **wifi** - Has wifi or not

### Output variable:

**price_range** - This is the target variable with value of 0 (cheap), 1 (mid priced), 2 (costly) and 3(expensive)
 
## Conclusion

During the analysis the given problem of predicting the price range we found that the problem is a multiclass classiffication problem and the dataset given is balanced with respect different categories of target feature (price_range).

We built different classifier for the given problem like KNN, Naive Bayes, Random Forest but Gradient Boosting Classifier outperformed other classifiers with test accuracy of 91.8% and a f1-score of 0.917.

Additionaly, while extracting feature importance from trained gradient boosting classifier we found that the feature RAM is the most important feature to predict price among all the features which is also true in practical scenario.

<hr>

## Model Performance

| Model |	Precision Score	| Recall Score | Accuracy Score	| f1-score |
:------------: | :------------: | :-------------: | :------------: | :-------------:
| Gradient Boosting	 |	0.917786	| 0.917883	| 0.918333	| 0.904119	| 0.917677 |

<hr>
