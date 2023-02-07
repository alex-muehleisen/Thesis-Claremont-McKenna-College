# 'How COVID-19 Impacted Tertiary Education: Predicting the Educational Infrastructure of the Future'
My Senior Thesis for a BA in Data Science at Claremont McKenna College. Includes an analysis of over 4,500,000 observations regarding tertiary education during COVID-19 and suggestions towards to the educational infrastructure of the future.

## Abstract:

In the following paper, data pertaining to educational shifts during the COVID-19 pandemic is gathered and analyzed. The pandemic shook the foundations of working life for many and the subsequent shift towards hybrid and work-from-home environments set the scene for perennial changes to the work-life balance. The impact of the COVID-19 pandemic on students is measured in this paper through literature review, data analysis, and regression analysis on student assessment scores. The pandemic shed light on the fact that education systems in the United States and around the world have not changed in hundreds of years. The rise of Zoom and other online-based educational resources has given way to new education models that oer hope for an improved educational infrastructure of the future. One of these models, mastery learning, is explored throughout this paper. The paper first identifies some of the existing problems with the implementation of mastery learning, before attempting to dispel these issues by analyzing current education models and the impact of COVID-19 on learning. Eventually, a educational framework for tertiary institutions of the future is suggested, and details of the framework are supported by findings from analysis conducted in this paper.

## Analysis Structure:

1. Multiple Linear Regression Analysis of Confidence on Student Performance
2. Multiple Linear Regression Analysis of External Factors on Student Performance
3. Predicting Student Performance through Classification
4. Trends in the impact of COVID-19 on Education
5. Random Forest Model on the Importance of factors in Online-Learning 

## Example Code (from analysis of External Factors on Student Performance):

### Initial Linear Model:

```
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
```

### Adding Polynomial Features:
```
polypipe_b = make_pipeline(PolynomialFeatures(4),LinearRegression())
polypipe_b.fit(Xbtrain, ybtrain)
polyscore_b = polypipe_b.score(Xbtest, ybtest)
```

### Adding Regularization:

```
polypipe2_b = make_pipeline(PolynomialFeatures(), Ridge())
grid_b = {'polynomialfeatures__degree':[1, 2, 3, 4, 5],'ridge__alpha':[.001,.01,.1, 1, 10, 100, 1000]}
search_b = GridSearchCV(polypipe2_b, grid_b)
search_b.fit(Xbtrain, ybtrain)
best_score_b = search_b.best_estimator_.score(Xbtest, ybtest)
best_degree_b = search_b.best_params_
```

### Predicting Scores through Classification:

```
kb = KBinsDiscretizer(n_bins=3,encode='ordinal')
ybins = kb.fit_transform(hsls_rna[['X1TXMTH']])[:,0]
```

### Adding a Support Vector Machine:

```
X2train,X2test,y2train,y2test = train_test_split(X_scaled, ybins, test_size=0.2, random_state=10)

svc = SVC(kernel='poly', degree=3, C=10)
svc.fit(X2train, y2train)
SVCscore = svc.score(X2test, y2test)
```
