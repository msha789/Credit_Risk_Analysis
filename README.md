# Credit_Risk_Analysis

An analysis using Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The purpose of this analysis is to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, focus was on **Supervised Learning** using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. 

# Results

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the loan status to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk". 

![image](https://user-images.githubusercontent.com/96354695/165444513-05b62e4c-8094-4e3a-ab45-1cc48d8b4ccf.png)

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.   

![image](https://user-images.githubusercontent.com/96354695/165444545-3e1e864f-cc06-4468-ac76-8a50ed223feb.png)

## Resampling Models to Predict Credit Risk

### Oversampling

RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

![image](https://user-images.githubusercontent.com/96354695/165444596-0da3b0de-8877-4d72-b538-b716efd540a6.png)

  * Balanced accuracy score: 64%.

  ![image](https://user-images.githubusercontent.com/96354695/165444622-f35c54b7-a689-41b7-98c4-e68b8dedf63c.png)

  * The "High Risk" precision rate was only 1% with the recall at 66% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and recall at 62%.  
  
  ![image](https://user-images.githubusercontent.com/96354695/165444704-9a30a6df-09fa-474c-b0ea-d1b61109cee2.png)
  
  ![image](https://user-images.githubusercontent.com/96354695/165444726-e7a3e7ab-ffd7-414a-928f-c787c1fbaddb.png)

SMOTE Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

  * The balanced accuracy score improved slightly to 65.1%.

  ![image](https://user-images.githubusercontent.com/96354695/165444756-aa4200d3-5e8a-4eb2-b23e-c9e27e5c1e6a.png)

  * Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and an improved recall at 69%.  

  ![image](https://user-images.githubusercontent.com/96354695/165444780-19f98517-d5f7-4753-8c66-92afecd16149.png)
  
  ![image](https://user-images.githubusercontent.com/96354695/165444807-e0c61477-934b-4bf4-9cfd-2ec8ff76f22d.png)

### Undersampling

ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

![image](https://user-images.githubusercontent.com/96354695/165444840-811f6b76-890c-4782-bc8a-c18d756bed3a.png)

  * Balanced accuracy score was lower than the oversampling models at 54.5%.

  ![image](https://user-images.githubusercontent.com/96354695/165444866-bef82adc-3174-4988-ab5a-cd05c4d1abe9.png)

  * The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
  * "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.  

  ![image](https://user-images.githubusercontent.com/96354695/165444900-a1b90d3d-ed75-40e7-9244-efd0d0685f8b.png)
  
  ![image](https://user-images.githubusercontent.com/96354695/165444922-18e87cc2-9f22-4114-9475-77659bc32d11.png)

## SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

SMOTEENN Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

![image](https://user-images.githubusercontent.com/96354695/165445000-d22e43ca-6fad-4388-b06d-01597d95d4aa.png)

  * The balanced accuracy score improved to 64.5% when using a combined sampling model.

  ![image](https://user-images.githubusercontent.com/96354695/165445047-4f91aa54-fa25-478b-a0ea-2cc52ff90db5.png)

  * The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
  * "Low Risk" still showed a precision rate of 100% with the recall at 57%.  
  
  ![image](https://user-images.githubusercontent.com/96354695/165445070-8188ab13-9d7e-4d8f-8a1b-df5e7f3da66c.png)

  ![image](https://user-images.githubusercontent.com/96354695/165445099-9f4eafb8-fae9-43fa-9b81-a57043c2099f.png)

## Ensemble Classifiers to Predict Credit Risk

Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

![Balancedcount](https://github.com/amylio/Credit_Risk_Analysis/blob/main/Images/balancedcount.png)

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class. 

  * The balanced accuracy score increased to 78.9% for this model.

  ![image](https://user-images.githubusercontent.com/96354695/165445144-2923fe10-822b-4ea3-8803-727288b2513c.png)

  * The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
  * "Low Risk" still had a precision rate of 100% with the recall at 87%.  
  * The top feature by importance was "total_rec_prncp" at 7.9% of the total.

  ![image](https://user-images.githubusercontent.com/96354695/165445177-b8920dbe-4d1f-434d-92fc-2d1d3be19193.png)
  
  ![image](https://user-images.githubusercontent.com/96354695/165445186-125446c9-71cf-45fc-9744-ad5b8c6861d5.png)

  ![image](https://user-images.githubusercontent.com/96354695/165445209-332684a5-28ee-4e62-a22d-af342f81f0b7.png)
  
EasyEnsembleClassifier Model, a set of classifiers where individual decisions are combined to classify new examples.

  * The balanced accuracy score increased to 93.2% with this model.

  ![image](https://user-images.githubusercontent.com/96354695/165445234-ed6a6e4d-23d8-4f83-b857-0e7370c51ca1.png)

  * The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
  * "Low Risk" still had a precision rate of 100% with the recall now at 94%.  

  ![image](https://user-images.githubusercontent.com/96354695/165445253-fd591e25-0950-48f3-8e9b-988c3ffd2a0d.png)
  
  ![image](https://user-images.githubusercontent.com/96354695/165445272-d41cb3d4-2d4e-4251-a1ae-e7ba6fc42678.png)

# Summary

In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the best choice.

