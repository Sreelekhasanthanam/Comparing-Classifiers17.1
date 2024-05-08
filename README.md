# Comparing-Classifiers17.1
<h1>Comparing-Classifiers</h1>
This python application using jupyter notebookm compares the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines classification models using dataset related to the marketing of bank products over the telephone.

[Jupiter Notebook used](https://github.com/Leopard-2019/Comparing-Classifiers/blob/main/notebook/PracticalApplicationAssignment_17.1-Copy1.ipynb)

</br>
The current CRISP-DM Process Model for Data Mining (see Figure 1) was followed.

</br>
</br>
<p align="center">
<img src="images/Figure1_CRISP_DM_Model.jpeg" width="300px" height="300px">
<h4 align="center"> Figure 1</h4>
</p>

<h2>Business Understanding</h2>
The Business goal is  to come up with the best machine learning classification model to predict if a future client will subscribe a term deposit or not based on several independent variables such as education level, marital status, if has housing loan or not, personal loan or not, etc. The best machine learning classification model is selected by ranking up four different machine learning models: KNeighborsClassifier, Logistic Regression, Support Vector Machine, and Decision Tree by their metrics and other indicators such as the Precision-recal curve, and confusion matrix. The dataset used to train those four models is related to the marketing of bank products over the telephone as mentioned before. The analysis was done using python & jupyter notebook.

<h2>Data Understanding</h2>
The original dataset (bank-full.csv) given is in .csv format.It consists of 17 columns and 6316 rows as shown below. The target/independent columns is "y" which is categorical (nominal feature), and it stands for: has the client subscribed a term deposit?. This variable is imbalance as will be later be seen. There are six numerical columns: 'age','balance','duration','previous','campaign','pdays' since the column: "duration" is only for benchmark purposes and is discarded for realistic predictive modelling, and the columns" day" and "month" were not considered numerical for obvious reason during the classification process..The rest of the columns are categorical (nominal). Most of the dataset provided is imbalanced before entering the modeling phase. None of the columns contain "NaN" values. Duplicates were not observed. It is thought that in order to provide more insight into the aforementioned dataset, a data preparation,i.e, data cleaning process needs to be done first.

</br>
</br>
<p align="center">
<img src="images/Figure17_2.jpeg" width="300px">
<h4 align="center"> Figure 2</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_3.jpeg" width="1000px">
<h4 align="center"> Figure 3</h4>
</p>


<h2>Data Preparation</h2>
The first step was check if there were any null values, and also make sure that there was not duplicates present in the dataset as well. As it is observed in Figure 4, there were initially no null values nor duplicates.

</br>
</br>
<p align="center">
<img src="images/Figure17_4.jpeg" width="800px">
<h4 align="center"> Figure 4</h4>
</p>

Columns: "job", "education", "poutcome", and "contact" have a feature with the same name: "unknown", so it was decided to replace it with different names to avoid potential problems in the foregoing analysis as indicated by Figure 5:

</br>
<p align="center">
<img src="images/Figure17_5.jpeg" width="800px">
<h4 align="center"> Figure 5</h4>
</p>


More insight into the dataset can be gained before finalizing the data preparation by doing the histograms for most of the categorical columns as shown on Figures 6, 7, and 8. All categorical columns/independent variables are nominal.

</br>
<p align="center">
<img src="images/Figure17_6.jpeg" width="1000px">
<h4 align="center"> Figure 6</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_7.jpeg" width="1000px">
<h4 align="center"> Figure 7</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_8.jpeg" width="1000px">
<h4 align="center"> Figure 8</h4>
</p>


<h3>Treatment of Outliers in Numerical Columns: "age", "balance", and "pdays"</h3>
The presence of outliers in the numerical columns: 'age','balance',,and 'pdays' were treated (see Figures 9, 10 and 11) indicated by the respective boxplot demands a careful and efective treatment before moving into the modeling phase. The histograms of the aforementioned columns have been also added for completeness.

</br>
</br>
<p align="center">
<img src="images/Figure17_9.jpeg" width="1000px">
<h4 align="center"> Figure 9</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_10.jpeg" width="1000px">
<h4 align="center"> Figure 10</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_11.jpeg" width="1000px">
<h4 align="center"> Figure 11</h4>
</p>

One pass was applied to the aforementioned columns in order to remove the outliers. The values equal to -1 was removed from the column "pdays". The aforementioned pass consisted on applying the well known Inter quartile range (IQR) method. Figure 12, 13, and 14 shows the final results after applying this method to remove the outliers. As it can be observed, this pass was very effective, i.e., removing the majority of the outliers, and improving the metrics during the modelling phase. As an additional comments, the target column "balance" shows a distribution skewed to the left, i.e, it was felt there was no need  to use its logarithm during modelling phase.

</br>
</br>
<p align="center">
<img src="images/Figure17_12.jpeg" width="1000px">
<h4 align="center"> Figure 12</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_13.jpeg" width="1000px">
<h4 align="center"> Figure 13</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_14.jpeg" width="1000px">
<h4 align="center"> Figure 14</h4>
</p>

The target column, i.e., the dependent variables: "y" is binary, and imbalanced as observed below:

</br>
</br>
<p align="center">
<img src="images/Figure17_y.jpeg" width="500px">
<h4 align="center"> Figure 15</h4>
</p>



Once an effective cleaning work has been completed, including removing most or all of the outliers. A boxplot "balance" vs. most of independent columns was built & analyized, indicating that in all categories: the client's age,client's job,client' marital status, client's education level, including housing's loan, personal's loan, even the  contact communication's, the clients that subscribed term deposit have more money in the their balance's account in general. Some of those plots are shown below:


</br>
</br>
<p align="center">
<img src="images/Figure17_job.jpeg" width="1000px">
<h4 align="center"> Figure 16</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_marital.jpeg" width="1000px">
<h4 align="center"> Figure 17</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_education.jpeg" width="1000px">
<h4 align="center"> Figure 18</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_housing.jpeg" width="1000px">
<h4 align="center"> Figure 19</h4>
</p>



<h4>Treatment of Categorical Features</h4>

<h3>Nominal Features</h3>
Nominal features are categorical features that have no numerical importance. Order does not matter. Most of the columns were found to fall in this category as follows: "job", "eudcation","contact","month", "day", "marital",and "poutcome". The Pandas getdummies function was used to creates dummy variables was used to treat them. A dummy variable is a numerical variable that encodes categorical information, having two possible values: 0 or 1. 
Those encoded features were added to the existing dataset using the panda function contact as shown  on Figure 20:

</br>
<p align="center">
<img src="images/Figure17_Nominal.jpeg" width="1000px">
<h4 align="center"> Figure 20</h4>
</p>

Binary data is also nominal data, meaning they represent qualitatively different values that cannot be compared numerically.There were three independent variables considered as binary: 'default' 'housing', ;'loan' all of them with 'yes' and 'no'.

</br>
<p align="center">
<img src="images/Figure17_binary.jpeg" width="1000px">
<h4 align="center"> Figure 21</h4>
</p>

<h3>Ordinal Features</h3>
None of the independent variables were considered to be treated as a ordinal feature, not even "poutcome", since there were a bunch of 'unknown' and 'other' items listed, beside 'failure' and 'success'.

</br>
</br>

Since, most of the columns have values between 0 or 1, it was decided to scale the column: "balance" as follow:

<p align="center">
<img src="images/Figure17_divided.jpeg" width="1000px">
<h4 align="center"> Figure 22</h4>
</p>

<h3>Splitting the variables</h3>
Splitting the dependent variable from the independent variables and assigning them to y and X respectively was done as follows:

</br>
<p align="center">
<img src="images/Figure17_splitting1.jpeg" width="1000px">
<h4 align="center"> Figure 23</h4>
</p>

The independent variables dataset is comprised by 77 columns and 6316 rows. Figure 24 shows the histogram for the columns comprising X dataset. Please keep in mind that the intention is not being able to see the label, just the bars, etc in the histogram, because there are too many histogram.

</br>
<p align="center">
<img src="images/Figure17_histo.jpeg" width="1000px">
<h4 align="center"> Figure 24</h4>
</p>

<h3>Cross-Validation Approach used</h3>
Although, the dependent variable is imbalanced, the HoldOut Cross-validation was used. In this technique, the whole dataset is randomly partitioned into a training set and validation set using the train_test_split function. The stratify parameter was used  to preserve  the dataset proportions for better prediction and reproduceability of results as follows:

</br>
</br>
<p align="center">
<img src="images/Figure17_splitting2.jpeg" width="1000px">
<h4 align="center"> Figure 25</h4>
</p>


<h2>Modelling</h2>
Although, working with imbalance data is always a challenge for any particular Machine Learning Model, four models were considered for the analysis: KNN, Logistic Regression, Support Machine Vector, and Decision Tree. The metric used to estimate the optimum parameters for each model was 'roc_auc' (the area under the ROC curve), since it works quite well for imbalance data.

<h4>KNeighborsClassifier</h4>
The supervised learning algorithm K-nearest neighbors (KNN) was used for classification in this analysis.The pipeline model is shown on Figure 26. 
                  
</br>
<p align="center">
<img src="images/Figure17_knn_1.jpeg" width="300px">
<h4 align="center"> Figure 26</h4>
</p>

The GridSearchCV function was used to optimized several hyper-parameters tested (see Figure 27). The optimum values for those, and the metrics results are shown on Figure 28. 

</br>
<p align="center">
<img src="images/Figure17_knn_2.jpeg" width="1000px">
<h4 align="center"> Figure 27</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_knn_44.jpeg" width="1000px">
<h4 align="center"> Figure 28</h4>
</p>

<h4>Logistic Regression</h4>
The supervised learning algorithm Logistic regression was also used for classification in this analysis, since the dependent variable is binary.The pipeline model is shown on Figure 29. 
                  
</br>
<p align="center">
<img src="images/Figure17_lgr_1.jpeg" width="300px">
<h4 align="center"> Figure 29</h4>
</p>

The GridSearchCV function was used to optimized several hyper-parameters tested (see Figure 30). The optimum values for those, and the metrics results are shown on Figure 31. 

</br>
<p align="center">
<img src="images/Figure17_lgr_2.jpeg" width="1000px">
<h4 align="center"> Figure 30</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_lgr_44.jpeg" width="1000px">
<h4 align="center"> Figure 31</h4>
</p>

<h4>Support Vector Machine</h4>
The supervised learning support vector machine (SVM) is used for classification in this analysis.The pipeline model is shown on Figure 32. 
                  
</br>
<p align="center">
<img src="images/Figure17_svc_1.jpeg" width="300px">
<h4 align="center"> Figure 32</h4>
</p>

The GridSearchCV function was used to optimized several hyper-parameters tested (see Figure 33). The optimum values for those, and the metrics results are shown on Figure 34. 

</br>
<p align="center">
<img src="images/Figure17_svc_2.jpeg" width="1000px">
<h4 align="center"> Figure 33</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_svc_44.jpeg" width="1000px">
<h4 align="center"> Figure 34</h4>
</p>

<h4>Decision Tree</h4>
The supervised learning decision tree is used for classification in this analysis.The pipeline model is shown on Figure 35. 
                  
</br>
<p align="center">
<img src="images/Figure17_tree_1.jpeg" width="300px">
<h4 align="center"> Figure 35</h4>
</p>

The GridSearchCV function was used to optimized several hyper-parameters tested (see Figure 36). The optimum values for those, and the metrics results are shown on Figure 37. 

</br>
<p align="center">
<img src="images/Figure17_tree_2.jpeg" width="1000px">
<h4 align="center"> Figure 36</h4>
</p>

</br>
<p align="center">
<img src="images/Figure17_tree_44.jpeg" width="1000px">
<h4 align="center"> Figure 37</h4>
</p>


<h2>Evaluation</h2>
As it can be observed,  the best model seems to be Logistic Regression as shown on Table 1. Also, it can be observed,  the elapsed time consumed for fitting for the SVC model is significantly larger than the rest of the models. The reason for that is that the option probability=True was used in this model, in order to be able to make it work to generate a later its precision- recall curve. This type of curve works much better for moderate to large imbalanced data than the ROC-curve, which is the case for the dataset used in this analysis. This curve ( see figure 38) may indicate that the best model is the Logistic Regression(red line), however,  it did consume a bit more elapsed time than that of KNeighborsClassifier. The confusion matrix (figure 39) and some of its associated statistics (see Table 2) suggest that the precision number seems to be a bit  better in the KNeighborsClassifier (73%) versus the Logistic regression  (67%), Support vector Machine  (63%),and Decision Tree (64%) , but Support Vector Machine model seems to have a bit better recall number (56%),  versus the Logistic regression  (53%), Decision Tree (54%),and  KNeighborsClassifier  (41%) . However, it is thought that the recall is more important, since the bank would greatly regret if potentially good clients are predicted not to subscribe a term deposit. In other words, minimizing the number of False Negatives would be of paramount importance for the bank, specially knowing that those outnumber the number of False Positives. However, overall, it was thought that Logistic Regression model may be considered the winner for classifying the dataset analyzed in this study. The KNeighborsClassifier and Decision Tree are fairly decent as well. 

</br>
</br>
<p align="center">
<img src="images/Figure17_eval_1_2.jpeg" width="500px">
<h4 align="center"> Table 1</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_eval_2.jpeg" width="600px">
<h4 align="center"> Figure 38</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_confusion.jpeg" width="1200px">
<h4 align="center"> Figure 39</h4>
</p>

</br>
</br>
<p align="center">
<img src="images/Figure17_eval_44.jpeg" width="400px">
<h4 align="center"> Table 2</h4>
</p>

<h2>Deployment</h2>
The code was written in Python, and it is available in a Jupyter Notebook that can be accessed in the link posted at the beginning of this document.

<h2>Main Conclusions & Recomendations</h2>
<p>1. The final dataset, after cleaning, removing outliers,  converting the categoricals data in 0 and 1 values,etc consists of 77 columns and 5304 rows. The target columns was "y" which stands: if a future client will subscribe a term deposit or not.</p>
<p>2. Overall, the best classification model is the Logistic Regression model for classifying the dataset analyzed in this study, however the other three models are fairly decent, in particular the KNeighborsClassifier, and Decision Tree models. </p>
<p> 3. It is important to highlight that the numerical variable "balance" was divided by 100, before initiating the regression modeling, since the majority of the columns have values 0 and 1. This helped to improved the metrics.</p>
<p> 4. There were five numerical independent variables used: 'age','balance','duration','previous','campaign','pdays', since "duration" was only for benchmark purposes and was discarded for realistic predictive modelling,the rest were nominals that were converted to values 0 and 1.  Therefore, most of the independent variables had values 0 and 1 used as final input during the modelling phase.</p>
<p> 5. The metric used to estimate the optimum parameters for each model was 'roc_auc', since it works quite well for imbalance data </p>
<p> 6. The precision- recall curve was chosen also as a indicator, since works much better for moderate to large imbalanced data than the ROC-curve, which is the case for the dataset used in this analysis.</p>
<p> 7. In all categories: the client's age,client's job,client' marital status, client's education level, including housing's loan, personal's loan, even the contact communication's, the clients that subscribed term deposit have more balance in their account in general. </p>
<p>8.It is recommend to test other classification models such as: Random Forest, Naive Bayes.
