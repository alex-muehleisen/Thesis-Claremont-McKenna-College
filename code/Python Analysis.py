# The following libraries were used in analysis:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Import Data

hsls_data = pd.read_csv('hsls_17_student_pets_sr_v1_0.csv')
hsls_data

# A summary of the data can be drawn with Pandas' `describe` function
# The function lists the number of observations, the mean, and the standard deviation for each column, 
# which provides a better understanding of the distribution of the data. 
# Minimum and maximum values are also noted, as well as interquartile ranges for each variable. 

summary1 = hsls_data[['X1MTHID', 'X1MTHEFF', 'X1MTHINT', 'X1TXMTH']].describe()
print(summary1)

# Next, the data and the feature matrix need to be checked for outliers. 
# This can be achieved using a `subplot` from *matplotlib* and the `boxplot` function from the *seaborn* package. 

warnings.filterwarnings('ignore')

fig, axs = plt.subplots(4, figsize = (5,5))
plt1 = sns.boxplot(hsls_data['X1MTHID'], ax = axs[0])
plt2 = sns.boxplot(hsls_data['X1MTHEFF'], ax = axs[1])
plt3 = sns.boxplot(hsls_data['X1MTHINT'], ax = axs[2])
plt4 = sns.boxplot(hsls_data['X1TXMTH'], ax = axs[3])

plt.tight_layout()
plt.show()

# The boxplots illustrate the number of extreme values present in the data, 
# so the next step should be to analyze each variable to find out what is going wrong. 
# Plotting the distributions with `displot` from *seaborn* will be helpful here. 

sns.displot(hsls_data['X1MTHID'], height=4)
sns.displot(hsls_data['X1MTHEFF'], height=4)
sns.displot(hsls_data['X1MTHINT'], height=4)
sns.displot(hsls_data['X1TXMTH'], height=4)
#plt.show()

# The plots above highlight the data clusters around values $-7$ and $-8$. 
# To note the significance of these values, the High School Longitudinal Study Codebook may be helpful. 
# The Codebook reveals that any non-response for a student in a given column is assigned a value of $-8$ for that observation. 
# Further, for columns **X1MTHEFF** and **X1MTHINT**, any student not taking a Math course in the Fall of 2009 is assigned a value of $-7$ for that respective column. 

# Using this information, the data cleaning process for these variables should be relatively simple. 
# Removing any observation below $-7$ from these columns is the only step that is necessary, 
# and it can be achieved with the following code chunk:

hsls_rna = hsls_data[(hsls_data['X1MTHID']>-7) & 
(hsls_data['X1MTHEFF']>-7) & 
(hsls_data['X1MTHINT']>-7) & 
(hsls_data['X1TXMTH']>-7)]

print(hsls_data.shape)
print(hsls_rna.shape)

# Now that the missing data has been removed, the relationship between the target column and the feature matrix 
# should be plotted to try to get a better picture of how the variables interact with each other. 
# The data explortation steps from above are repeated to do this.

summary2 = hsls_rna[['X1MTHID', 'X1MTHEFF', 'X1MTHINT', 'X1TXMTH']].describe()
print(summary2)

warnings.filterwarnings('ignore')

fig, axs = plt.subplots(4, figsize = (5,5))
plt10 = sns.boxplot(hsls_rna['X1MTHID'], ax = axs[0])
plt20 = sns.boxplot(hsls_rna['X1MTHEFF'], ax = axs[1])
plt30 = sns.boxplot(hsls_rna['X1MTHINT'], ax = axs[2])
plt40 = sns.boxplot(hsls_rna['X1TXMTH'], ax = axs[3])

plt.tight_layout()
plt.show()

sns.displot(hsls_rna['X1TXMTH'])
#plt.show()

# The normality of the data now that the missing values have been removed is also important to note here. 
# This should be expected, given the nature of the **X1TXMTH** variable. 
# Referring to the HSLS Codebook, this information is 'norm-referenced,' meaning it gives an estimate of achievement relative to the population. 
# Given this information, a normally-distributed plot for this variable is expected, 
# and can be seen a lot more clearly when the extreme values are removed from the data.

##### Heatmap

# A correlogram can reveal the correlation between each of our identified variables by 
# displaying a 2D correlation matrix between selected variables. 
# The plot provides a clear visualization of the strength of relationships between variables. 
# A score of '1' inside of the correlation matrix means that there is a perfect, positive relationship between 
# two variables on its axes, whereas a score of '0' would indicate that there is no relationship between the variables.

warnings.filterwarnings('ignore')

sns.heatmap(hsls_rna[['X1MTHID', 'X1MTHEFF', 'X1MTHINT', 'X1TXMTH']].corr(), annot = True)
plt.rcParams["figure.figsize"] = (7,10)
plt.show()

# Overall, an important feature of the heatmap to note is that all of the measured correlations are positive and below a score of 0.5. 
# The reason for this may become clearer after plotting the relationships between the feature variables and the target variables. 
# The `pairplot` function from the *seaborn* package can be used for this. 

sns.pairplot(hsls_rna, x_vars=['X1MTHID', 'X1MTHEFF', 'X1MTHINT'],
y_vars='X1TXMTH', kind='scatter')
#plt.show()

##### Initial Linear Model

# First, the identified predictor variables are extracted from the table and assigned to variable `X`. 
# Similarly, the target variable is extracted from the table and assigned to variable `y`. 
# The `StandardScalar` function from the *scikit-learn* library is used to transform the data to 
# follow a Standard Normal Distribution, making the mean equal to 0 and scaling the data to unit variance. 
# Next, the `train_test_split` function is used to split the data into a training and a testing set, 
# using a test size of 20 percent of the data. The model will be trained using the training set and tested 
# against the testing set to measure its predictive accuracy. The `LinearRegression` function from *scikit-learn* 
# will be used to create the linear model and fit it to the training data. 

X = hsls_rna[['X1MTHID', 'X1MTHEFF', 'X1MTHINT']]
y = hsls_rna[['X1TXMTH']]
X_scaled = StandardScaler().fit_transform(X)
Xtrain,Xtest,ytrain,ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=10)
lin_mod = LinearRegression()
lin_mod.fit(Xtrain, ytrain)
lin_mod_score = lin_mod.score(Xtest, ytest)
lin_mod_score
coefficients = lin_mod.coef_
coefficients

##### Adding Polynomial Features

# The created linear model is not very accurate at all right now, but it could potentially be improved by adding polynomial features. 
# This is done by raising existing features to an exponent to allow the model to identify nonlinear patterns in the data. 
# Since `lin_mod_score` is not particularly strong, adding such features could help improve the model significantly. 
# In this case, a pipeline is created using *scikit-learn's* `make_pipeline` function. 
# The pipe adds cubic features (polynomial features of degree 3) and then fits a Linear Regression Model. 
# Its accuracy is measured by the `polyscore` variable. 

polypipe = make_pipeline(PolynomialFeatures(4),LinearRegression())
polypipe.fit(Xtrain, ytrain)
polyscore = polypipe.score(Xtest, ytest)
polyscore

##### Adding Regularization

# To further improve the model, regularization can be added. 
# Regularization is a method to avoid overfitting the data. 
# By adding regularization, the coefficient estimates are constrained so to be shrunk towards zero. 
# It reduces the complexity of the regression, thus avoiding overfitting. 

# A new Polynomial Pipeline should be created, but this time with an unspecified degree. 
# The pipe should also include *skikit-learn's* `Ridge` function, which is a Linear Regressor with L2 Regularization. 
# Additionally, a Grid Search can be set up to test various values to find the best value for the model's degree and alpha components. 

polypipe2 = make_pipeline(PolynomialFeatures(), Ridge())
grid = {'polynomialfeatures__degree':[1, 2, 3, 4, 5],'ridge__alpha':[.001,.01,.1, 1, 10, 100]}
search = GridSearchCV(polypipe2, grid)
search.fit(Xtrain, ytrain)
best_score = search.best_estimator_.score(Xtest, ytest)
best_score
best_degree = search.best_params_
best_degree

##### The Influence of External Factors on Student Performance

# Since the linear regression of confidence on student performance did not yield favorable results, 
# the next step in the analysis is to run a Multiple Linear Regression on three new variables: 
# **X1FAMINCOME**, **X1DUALLANG**, and **X1TMEFF**, which measure the Total Family Income from all Sources in 2008, 
# whether a student's first language was only-English or English and an additional language, 
# and the scale of the math teacher's self-efficacy, respectively. 

# Again, the `describe` function from Pandas is used to compute descriptive statistics for the feature matrix and the target column. 

summary3 = hsls_data[['X1FAMINCOME', 'X1DUALLANG', 'X1TMEFF', 'X1TXMTH']].describe()
print(summary3)

# The data needs to be checked for outliers again. 
# The same process of plotting the distribution of the variables will be 
# undertaken using the same functions from *matplotlib* and *seaborn*. 
# Since the Math Theta Score column (**X1TXMTH**) has already been cleaned, it will be left out of this process.

warnings.filterwarnings('ignore')

fig, axs = plt.subplots(3, figsize = (5,5))
plt5 = sns.boxplot(hsls_data['X1FAMINCOME'], ax = axs[0])
plt6 = sns.boxplot(hsls_data['X1DUALLANG'], ax = axs[1])
plt7 = sns.boxplot(hsls_data['X1TMEFF'], ax = axs[2])

plt.tight_layout()
plt.show()

sns.displot(hsls_data['X1FAMINCOME'], height=4)
sns.displot(hsls_data['X1DUALLANG'], height=4)
sns.displot(hsls_data['X1TMEFF'], height=4)
#plt.show()

# The plots above provide a solid base to work from to start the cleaning process. 
# They reveal that like our previous variable set, all of these column place missing 
# and non-response observations at a value below -7. 
# We can confirm this by checking the HSLS Online Codebook again. 
# After doing so, the data cleaning process for this feature matrix will end up looking quite similar to the last cleaning process. 

hsls_rna2 = hsls_data[(hsls_data['X1FAMINCOME']>-7) & 
(hsls_data['X1DUALLANG']>-7) & 
(hsls_data['X1TMEFF']>-7) & 
(hsls_data['X1TXMTH']>-7)]

print(hsls_data.shape)
print(hsls_rna.shape)
print(hsls_rna2.shape)

# Now that the missing data has been removed, a better picture of the relationship 
# between the target column and the feature matrix can be mapped out. 

summary4 = hsls_rna2[['X1FAMINCOME', 'X1DUALLANG', 'X1TMEFF', 'X1TXMTH']].describe()
print(summary4)

sns.displot(hsls_rna2['X1FAMINCOME'], height=4)
sns.displot(hsls_rna2['X1DUALLANG'], height=4)
sns.displot(hsls_rna2['X1TMEFF'], height=4)
#plt.show()

##### Heatmap

# Reviewing the heatmap again for the new feature matrix is a great place to start before diving into analysis. 

warnings.filterwarnings('ignore')

sns.heatmap(hsls_rna2[['X1FAMINCOME', 'X1DUALLANG', 'X1TMEFF', 'X1TXMTH']].corr(), annot = True)
plt.rcParams["figure.figsize"] = (7,10)
plt.show()

##### Multiple Linear Regression of External Factors on Student Performance

# The linear regression process will now be run again, this time estimating the effect of external factors, 
# such as financial situation, background, and teacher expectations, on student performance. 

##### Initial Linear Model

# First, the identified predictor variables are extracted from the table and assigned to variable `Xb`. 
# Similarly, the target variable is extracted from the table and assigned to variable `yb`. 
# The `StandardScalar` function and the `train_test_split` function from the *scikit-learn* library 
# are both again used to normalize the data and then split it into training and testing sets. 
# The model will be trained using the training set and tested against the testing set to measure its predictive accuracy. 
# The `LinearRegression` function from *scikit-learn* will again be used to create the linear model and fit it to the training data. 

Xb = hsls_rna2[['X1FAMINCOME', 'X1DUALLANG', 'X1TMEFF']]
yb = hsls_rna2[['X1TXMTH']]
Xb_scaled = StandardScaler().fit_transform(Xb)
Xbtrain,Xbtest,ybtrain,ybtest = train_test_split(Xb_scaled, yb, test_size=0.2, random_state=10)
lin_mod_b = LinearRegression()
lin_mod_b.fit(Xbtrain, ybtrain)
lin_mod_b_score = lin_mod_b.score(Xbtest, ybtest)
lin_mod_b_score
coefficients_b = lin_mod_b.coef_
coefficients_b

##### Adding Polynomial Features

# The model is not very accurate, but it looks like it could be relying heavily on the **X1FAMINCOME** column. 
# It might be able to be improved by adding polynomial features. *Scikit-learn's* `make_pipeline` function can be used again 
# here to add polynomial features (again of degree 3) and then fit a Linear Regression Model. 
# Its accuracy is measured by the `polyscore_b` variable. 

polypipe_b = make_pipeline(PolynomialFeatures(4),LinearRegression())
polypipe_b.fit(Xbtrain, ybtrain)
polyscore_b = polypipe_b.score(Xbtest, ybtest)
polyscore_b

##### Adding Regularization

# To further improve the model, regularization can be added. 
# Again, a new Polynomial Pipeline will be created with unspecified degree and with 
# the L2 Regularization of *skikit-learn's* `Ridge` function. 
# The Grid Search will also be set up again to test various values for the model's best degree, 
# as well as the Ridge's optimal alpha.  

polypipe2_b = make_pipeline(PolynomialFeatures(), Ridge())
grid_b = {'polynomialfeatures__degree':[1, 2, 3, 4, 5],'ridge__alpha':[.001,.01,.1, 1, 10, 100, 1000]}
search_b = GridSearchCV(polypipe2_b, grid_b)
search_b.fit(Xbtrain, ybtrain)
best_score_b = search_b.best_estimator_.score(Xbtest, ybtest)
best_score_b
best_degree_b = search_b.best_params_
best_degree_b

##### Predicting Mathematics Scores through Classification

# The regressions just run can be turned into a classification problem using `KBinsDiscretizer` from *skikit-learn*. 
# This function creates a new variable, `y_bins`, which contains a value (2, 1, or 0) reflecting whether 
# the Math Assessment Score for a particular student falls into the top, middle, or bottom third of all test score observations in the data.

kb = KBinsDiscretizer(n_bins=3,encode='ordinal')
ybins = kb.fit_transform(hsls_rna[['X1TXMTH']])[:,0]

##### Adding a Support Vector Machine

# A Support Vector Machine (SVM) is a flexible supervised machine learning method used for 
# both regression and classification. In this analysis, it will be used for the latter. 
# A Linear Support Vector Classifier (SVC) [`LinearSVC`] will be used to fit the data to return a 
# best-fit 'hyperplane,' which essentially categorizes the data. 

# Splitting the new data with `y_bins` into training and testing sets, 
# the next step is to set up a new model to analyze the data for classification analysis. 
# This can be achieved with a Support Vector Machine from *skikit-learn*, using a polynomial kernel. 
# Initial analysis will be trialed with an SVC with a degree of 2 and a level of regularization ($C$) of 10. 

X2train,X2test,y2train,y2test = train_test_split(X_scaled, ybins, test_size=0.2, random_state=10)

svc = SVC(kernel='poly', degree=3, C=10)
svc.fit(X2train, y2train)
SVCscore = svc.score(X2test, y2test)
SVCscore

##### Logistic Regression

# Logistic regression is another good option for classification problems, and *skikit-learn's* `LogisticRegression` 
# function can be used to achieve this. With the help of this function, a predictive analysis 
# will be run to explain the relationship between the three independent variables and the target variable. 
# Ultimately, the probabilities behind these relationships will be estimated using a logistic regression equation. 

logr = LogisticRegression()
logr.fit(X2train, y2train)
logrscore = logr.score(X2test, y2test)
logrscore

##### Confusion Matrix

# Constructing a confusion matrix for classification problems can be extremely beneficial 
# to look at how the trained model predicts the testing data. 
# It can be constructed using the `confusion_matrix` function from *skikit-learn*. 

confusion_matrix(y2test, logr.predict(X2test))

# The diagonals of the confusion matrix display true positives (testing values that the model correctly predicted), 
# while the values in the corners represent false positives (cases in which the model incorrectly predicted the test data). 
# It seems like the model did fairly well overall, but did a lot better at predicting assessment scores 
# in the top and bottom thirds, as opposed to the middle third. 

##### Simple Linear Regression

# Out of all of the feature variables analyzed, the best predictor within the 
# respective interest clusters seemed to be **X1TXMTH**. 
# To push the analysis towards identifying a single variable that can explain excellent student 
# performances and thus simplify the study, this variable will be regressed on further. 

# A Simple Linear Regression with one predictor variable and one target variable is a great option for further analysis.

X3 = hsls_rna[['X1MTHID']]
y3 = hsls_rna['X1TXMTH']
lin_mod3 = LinearRegression()
lin_mod3.fit(X3,y3)
plt.scatter(hsls_rna[['X1MTHID']],y3)
RSS = ((y3-lin_mod3.predict(X3))**2).sum()
R2score = lin_mod3.score(X3,y3)
RSS, R2score


