<picture><img src="https://akm-img-a-in.tosshub.com/businesstoday/images/story/202202/phishing-1200-sixteen_nine.jpg?size=948:533" alt="hacker-1" width="4000" height="260"></picture>
<br><br>
# Web Phishing Detection 
IBM-Project-34397-1660235048


**Batch Name :** B2-2M4E

**Team ID :** PNT2022TMID15077

## Team Members :  
   
  - Shrivas S
  - Shanmugam S
  - Ramanaathan S 
  - Rahul B

## Table of Content
  * [Project Objectives](#project-objectives)
  * [Introduction](#introduction)
  * [Common threats of web phishing](#common-threats-of-web-phishing)
  * [Technical Architecture](#technical-architecture)
  * [Result](#result)
    <br>  
     - [Accuracy of various models](#accuracy-of-various-models)
    <br>
     - [Feature importance for Phishing URL Detection](#feature-importance-for-phishing-url-detection)
  * [Conclusion](#conclusion)
  * [Technologies Used](#technologies-used)

 # Project Objectives  
 

 #### By the end of this project :
 - A phishing website is a common social engineering method that mimics trustful uniform resource locators(URLs)and webpages.
 - The objective of this project is to train machine learning models and deep neural nets on the dataset created to predict phishing websites.
 - Both phishing and benign URLs of websites are gathered to form a dataset and from the required URL and website content-based features are extracted.

  - The performance level of each model is measures and compared.

 # Introduction 
  Phishing is the most commonly used social engineering and cyber attack.
  Through such attacks,the phisher targets naive online users by tricking them into revealing confidential information,with the purpose of using it fraudulently.
  In order to avoid getting phished,
  1. Users should have awareness of phishing websites.
  2. Have a blacklist of phishing websites which requires the knowledge of website being detected as phishing.
  3. Detect them in their early appearance ,using machine leaning and deep neutral network algorithms.
  Of the above three,the machine learning based method is proven to be most effective than the other methods.
<br><br>
# Common threats of web phishing 

1.  Web phishing aims to steal private information, such as usernames, passwords, and credit card    details, by way of impersonating a legitimate entity.

2.  Large organizations may get trapped in different kinds of scams.

3. Email phishing is the most common type of phising and it has been in use since the 1990s.

4.  This Guided Project mainly focuses on applying a machine-learning algorithm to detect Phishing websites.

>In order to detect and predict e-banking phishing websites, we proposed an intelligent, flexible and effective system that is based on using classification algorithms. We implemented classification algorithms and techniques to extract the phishing datasets criteria to classify their legitimacy. The e-banking phishing website can be detected based on some important characteristics like URL and domain identity, and security and encryption criteria in the final phishing detection rate. Once a user makes a transaction online when he makes payment through an e-banking website our system will use a data mining algorithm to detect whether the e-banking website is a phishing website or not.
<br><br>
# Technical Architecture  
  ![pasted image 0](https://user-images.githubusercontent.com/62200224/191585875-9db35871-72b5-476e-ac9b-3795cf3778de.png)
  <br><br>

# Result

## Accuracy of various models

||ML Model|	Accuracy|  	f1_score|	Recall|	Precision|
|---|---|---|---|---|---|
1|	CatBoost Classifier	|0.972	|0.976	|0.994	|0.987|
2|	Gradient Boosting Classifier	|0.971	|0.975	|0.992	|0.985|
3|	Random Forest	|0.967	|0.972	|0.994	|0.986|
4|	Decision Tree|	0.961|	0.965|	0.992|	0.991|
5|	Support Vector Machine	|0.957	|0.963	|0.982|	0.966|
6|	K-Nearest Neighbors|	0.944	|0.950|	0.962|	0.996|
7|	Logistic Regression	|0.924	|0.933	|0.947	|0.927|
8|	Naive Bayes Classifier|	0.583|	0.420|	0.291|	0.996|

## Feature importance for Phishing URL Detection 
[<img src="https://i.ibb.co/4RTHNnZ/123.png" alt="123" border="0">]()
<br><br>

# Conclusion
1. The final take away form this project is to explore various machine learning models, perform Exploratory Data Analysis on phishing dataset and understanding their features. 
2. Creating this notebook helped me to learn a lot about the features affecting the models to detect whether URL is safe or not, also I came to know how to tuned model and how they affect the model performance.
3. The final conclusion on the Phishing dataset is that the some feature like "HTTTPS", "AnchorURL", "WebsiteTraffic" have more importance to classify URL is phishing URL or not. 
4. CatBoost Classifier currectly classify URL upto 97.2% respective classes and hence reduces the chance of malicious attachments.
<br><br>
 # Technologies Used

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" width=200>](https://numpy.org/doc/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" width=200>](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
[<img target="_blank" src="https://images.velog.io/images/seokbin/post/e14f498a-a0b1-4880-9a88-98be38c50267/jupyter_logo_icon_169453.png " width=200>](https://jupyter.org/) 
[<img target="_blank" src="https://kevin-hartman.io/images/tech/scikit-learn-logo.png" width=200>](https://scikit-learn.org/stable/) 
[<img target="_blank" src="https://cdn.icon-icons.com/icons2/2699/PNG/512/pocoo_flask_logo_icon_168045.png" width=200>](https://flask.palletsprojects.com/en/2.0.x/) 
[<img target="_blank" src="https://financialit.net/sites/default/files/ibm_cloud-ar21_0.png" width=200>](https://www.ibm.com/in-en/cloud?utm_content=SRCWW&p1=Search&p4=43700052661371387&p5=e&gclid=CjwKCAiAvK2bBhB8EiwAZUbP1MNb2RC3e1TDBJhMnJdNAtW-tlFA3uNoTQQwUu0EKRW6GeB9INwZ4BoCpLgQAvD_BwE&gclsrc=aw.ds) 
