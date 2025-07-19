# Employee-Salary-Prediction


Employee Salary Prediction using Machine Learning

Overview

This project uses machine learning algorithms to predict employee salaries based on various features. The project is implemented in Python using Jupyter Notebook and utilizes popular libraries such as scikit-learn, pandas, and NumPy.

Features

- Data Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
- Model Selection: Comparison of multiple machine learning algorithms, including Linear Regression, Decision Trees, Random Forest, and Support Vector Machines.

  Machine Learning Algorithms accuracy scores:
   -LogisticRegression
  # machine learning algorithm
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)#input and output training data
predictl=lr.predict(xtest)          
predictl
from sklearn.metrics import accuracy_score
accuracy_score(ytest,predictl)
0.8219637430577387
  -MLPClassifier
  # machine learning algorithm
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver ='adam',hidden_layer_sizes=(5,2),random_state=2,max_iter=2000)
clf.fit(xtrain,ytrain)#input and output training data
predict2=lr.predict(xtest)          
predict2
from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict2)
0.8219637430577387


  -KNeighborsClassifier
  # machine learning algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(xtrain,ytrain)#input and output training data
predict=knn.predict(xtest)          
predict
from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict)
0.8240595200670648
Requirements

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- NumPy
- Matplotlib

Conclusion

This project demonstrates the application of machine learning algorithms to predict employee salaries.


