##  HOUSE PRICING MODEL & PERFORMANCE VISUALIZATION

**Dataset Used**: AmesHousing.csv dataset from Kaggle, containing details about houses sold in the US. The target value is Price.

* Chosen Coulmns:
* Lot Area (Rozmiar działki)
* Gr Liv Area (Rozmiar domu nad ziemią)
* Bldg Type (typ domu)
* House Style (styl domu)
* Overall Qual (Ogólna jakość)
* Year Built (Rok budowy)
* TotRms AbvGrd (Liczba pokoi nad ziemią)
* Garage Cars (Pojemność garażu w liczbie aut)
* Neighborhood (Dzielnica)
* Total Bsmt SF (Całkowita powierzchnia piwnicy)
* Kitchen Qual (Jakość kuchni)
* Year Remod/Add (Rok przebudowy / generalnego remontu)

**The project consists of 2 main parts**:
1. Data cleaning, exploratory data analysis (EDA), and building a machine learning model in Python.
2. Visualization of model performance and key KPIs in Power BI.

### Python Part contains:
1. Basic EDA (Exploratory Data Analysis) to understand data distribution.
<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/e45aa060-9d7d-488e-b7e7-884e3237512e" />
<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/fe68fb3f-60b5-4c1f-b25c-5675e7e4e8d2" />
<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/0a33e580-b6a7-4048-a39b-ab9c99af31c6" />
<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/097837b6-5eae-460c-9704-c49896c0f12f" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/7cd7031d-46df-414c-8b74-be6669265467" />
<img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/d1cab0d2-6783-4888-b517-0da6d1035ebc" />

2. Data Cleaning & Feature Mapping.
3. Creating Machine Learning Pipelines (using Imputers, Scalers, Encoders, and ColumnTransformer).
4. Performing Cross-Validation (GridSearchCV) to evaluate 3 models: Ridge, XGBoost, and LinearSVR.
5. Applying a logarithmic transformation to the target variable.
6. Model fitting, predicting, and calculating evaluation errors.
7. Checking feature importances

### Power BI Part contains:
* KPI Cards: Displaying main model evaluation metrics (MAE, MAPE, RMSE, Bias, and Max Error).
* Residual Plot: Visualizing model errors to diagnose heteroskedasticity and identify outlier predictions.
* Feature Importance Chart: Highlighting the most impactful variables driving the model's decisions.
* Error vs. Features Analysis: Custom charts comparing error KPIs across specific categories (e.g., Neighborhood, Garage Cars).
* Interactive Play Axis: Animating the timeline (Year Built) to dynamically update all dashboard visuals and observe performance trends over time.

<img width="1097" height="626" alt="image" src="https://github.com/user-attachments/assets/b1228a2e-36b1-467a-8ae7-971a7910e83b" />

👀 Check Raport video here:
https://github.com/user-attachments/assets/e287e312-c129-4ea3-a0f7-4e6cefc56419





