# Global Climate Change Analysis

*The dataset for this project originates from the [Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data).*

*Notebook published on Anaconda. [click here](https://anaconda.org/arun2728/globalclimatechangeanalysis/notebook)*

## Background
Some say climate change is the biggest threat of our age while others say it’s a myth based on dodgy science. We are turning some of the data over to you so you can form your own view.

### Problem Statement:
In this problem we have to perform in-depth analysis to study the change of climate across all many years. Also we have to build a model capable enough to forecast temperature of india.

## Dataset
The dataset originates from the Berkeley Earth Surface Temperature Study. It combines 1.6 billion temperature reports from 16 pre-existing archives. It is nicely packaged and allows for slicing into interesting subsets (for example by country). They publish the source data and the code for the transformations they applied. They also use methods that allow weather observations from shorter time series to be included, meaning fewer observations need to be thrown away.

In this dataset, we have several files:

* GlobalTemperatures.csv - Global Land and Ocean-and-Land Temperatures

* GlobalLandTemperaturesByCountry.csv - Global Average Land Temperature by Country

* GlobalLandTemperaturesByState.csv - Global Average Land Temperature by State

* GlobalLandTemperaturesByMajorCity.csv - Global Land Temperatures By Major City

* GlobalLandTemperaturesByCity.csv - Global Land Temperatures By City

## Model Performance

| Model |	Mean Squared Error		| Root Mean Squared Error | 
:------------: | :------------: | :-------------: | 
| Seasonal-ARIMA		 |	0.2421	| 0.492036584	|

## Model Inferences

![no image](https://github.com/arun2728/data-science-portfolio/blob/main/Global%20Climate%20Change/output/inference.png)

<hr>

## Conclusion

During my research it was found that there has been a global increase trend in temperature, particularly over the last 30 years. This is due to the violent activities of a humankind. In more developed countries the temperature began to register much earlier. Over time the accuracy of the observations is increased, that is quite natural. Mankind must reflect and take all necessary remedies to reduce emissions of greenhouse gases in the atmosphere.

Additionally, I have build a Seasonal-ARIMA model to forecast temperature of Bomaby city. The built model is than used to predict the temperature of bombay for year 2013.


#### Model Forecasting on temperature of Bombay in year 2013

![no image](https://github.com/arun2728/data-science-portfolio/blob/main/Global%20Climate%20Change/output/Forecast.png)

According to the forecasting Bombay will record a highest temperature of **28.55ºC** in the month of April i.e during summers. Additionaly, monsoon is going to be cooler and there will increase in temperature in post-monsoon period. The temperature in winter's will remain same i.e **25ºC**.
