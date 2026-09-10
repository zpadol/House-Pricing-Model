import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import TransformedTargetRegressor

df = pd.read_csv("AmesHousing.csv")


pd.set_option('display.expand_frame_repr', False)
print(df.head())
print(df.describe())
print(df.columns)


"""
Lot Area (Rozmiar działki)
Gr Liv Area (Rozmiar domu nad ziemią)
Bldg Type (typ domu)
House Style (styl domu)
Overall Qual (Ogólna jakość)
Year Built (Rok budowy)
TotRms AbvGrd (Liczba pokoi nad ziemią)
Garage Cars (Pojemność garażu w liczbie aut)
Neighborhood (Dzielnica)
Total Bsmt SF (Całkowita powierzchnia piwnicy)
Kitchen Qual (Jakość kuchni)
Year Remod/Add (Rok przebudowy / generalnego remontu)
"""
num = ['Lot Area', 'Gr Liv Area','TotRms AbvGrd', 'Year Built', 'Year Remod/Add', 'Garage Cars','Total Bsmt SF','Overall Qual']
cat = ['Bldg Type', 'House Style', 'Kitchen Qual']
df_num = df[num]

df_num.hist(figsize=(15, 12), bins = 20, color = 'steelblue')
plt.show()
# if using linear model, logaithming the variable 'Lot Area' will be needed (long tail).

plt.figure(figsize=(15, 12))
for i, kol in enumerate(num,1):
    plt.subplot(3,3,i)
    sns.scatterplot(
        data = df,
        x = kol,
        y = 'SalePrice',
        color = 'steelblue',
        alpha = 0.4,
        size=8,
        legend = False
    )
    plt.title(f'{kol} vs Price')
plt.tight_layout()
plt.show()

"""
Sale Price vs. Numerical Features
Property Size & Area (Gr Liv Area, Total Bsmt SF, Lot Area)
Both above-ground living area and basement size show a strong, stable positive correlation with the sale price, making them key predictive features. While the lot area also increases the price rapidly at first, it quickly plateaus; notably, all these area charts expose two massive outliers (huge properties sold for cheap) that must be filtered out.

Capacity Metrics (TotRms AbvGrd, Garage Cars)
There is a clear upward trend where an increase in the number of rooms and garage capacity positively impacts the house value. The apparent price drop for massive houses (12-14 rooms) or large garages (4-5 cars) is strictly an illusion caused by a very small sample size for these extreme values.

Age and Renovations (Year Built, Year Remod/Add)
Historically older homes maintain a relatively stable, moderate baseline price without significant fluctuations. However, there is a sharp, exponential price premium for newly built or very recently renovated properties.

Quality Assessment (Overall Qual)
The overall material and finish quality demonstrates a textbook, almost perfect linear relationship with the sale price. It stands out as the strongest individual indicator in the dataset, with prices stepping up predictably and significantly for each higher quality tier.
"""

plt.figure(figsize=(15, 5))
for i,kol in enumerate(cat,1):
    plt.subplot(1,3,i)
    sns.countplot(
        data = df,
        x = kol,
        color = 'lightgreen'
    )
    plt.title(f'{kol} count')
    plt.xticks(fontsize=6)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
for i,kol in enumerate(cat,1):
    plt.subplot(1,3,i)
    sns.boxplot(
        data = df,
        x = kol,
        y = 'SalePrice',
        color = 'lightgreen'
    )
    plt.title(f'{kol} vs Price')
    plt.xticks(fontsize=6)
plt.tight_layout()
plt.show()

"""
Building Type (Bldg Type)
Single-family detached homes (1Fam) and end-unit townhouses (TwnhsE) command the highest market prices, reflecting a strong buyer preference for privacy and space. In contrast, multi-dwelling setups like two-family conversions (2fmCon) and standard duplexes (Duplex) consistently represent the most budget-friendly properties in the dataset.

House Style and Finish (House Style)
Properties maximizing usable vertical space, specifically fully finished two-story (2Story) and two-and-a-half-story (2.5Fin) houses, achieve the highest valuations. Notably, there is a massive price penalty for unfinished half-stories; houses with completed upper levels (e.g., 1.5Fin, 2.5Fin) sell for significantly more than their unfinished counterparts (1.5Unf, 2.5Unf).

Kitchen Quality (Kitchen Qual)
Kitchen condition is a major driver of overall property value, showing a strict pricing hierarchy. An Excellent (Ex) kitchen yields the highest premium by far, followed predictably by Good (Gd), Typical/Average (TA), Fair (Fa), and Poor (Po) (with little observations).
"""


corr = df_num.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr, mask=mask, cmap='Blues', square=True, annot=True, fmt='.2f')
plt.show()

"""
there are some variables with high correlation. If using linear model should consider deleting one of them. 
However I will probably use tree-based model so high correlations are not problematic. 
"""

# Additionally - Business insight
plt.figure(figsize=(10,8))
sns.boxplot(data = df, x = 'Garage Cars', y = 'Year Built', palette = 'magma', hue  = 'Garage Cars')
plt.show()


# next step - data cleaning

# dropping data that looks "suspicious"
df = df[(df['Lot Area'] <150_000) | df['Lot Area'].isna()]
df = df[(df['Gr Liv Area'] <4500) | df['Gr Liv Area'].isna()]
df = df[(df['Total Bsmt SF'] <4000) | df['Total Bsmt SF'].isna()]

# Capping rare values
df.loc[df['Garage Cars']  > 3 ,'Garage Cars'] = 3
df.loc[df['TotRms AbvGrd'] > 12, 'TotRms AbvGrd'] = 12

print(df['House Style'].value_counts())
df.loc[df['House Style'].str.contains('Unf') | df['House Style'] == '2.5Fin'] = 'Other'

print(df['Bldg Type'].value_counts())
# leaving as it is

print(df['Kitchen Qual'].value_counts())
df.loc[df['Kitchen Qual'] == 'Po', 'Kitchen Qual'] = 'Fa'

print(df['Neighborhood'].value_counts())
top = df['Neighborhood'].value_counts().head(15).index
df.loc[~df['Neighborhood'].isin(top), 'Neighborhood'] = 'Other'

all_col = num + ['Bldg Type', 'House Style', 'Kitchen Qual', 'Neighborhood']

X = df[all_col]
y = df['SalePrice'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Numeric pipeline will contain : Simple imputer (median), Standard Scaler(in case of using linear-based model)
# Categorical pipeline will contain (without Kitchen and Neighborhood: Simple imputer (most_frequent), one hot encoder
# Kitchen pipeline will contain : Simple imputer (most_frequent), Ordinal Encoder
# Neighborhood pipeline will contain:  Simple imputer (most_frequent), Target Encoder

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ( 'scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

Kitchen_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('oe', OrdinalEncoder(categories=[['Fa', 'TA', 'Gd', 'Ex']], handle_unknown='use_encoded_value', unknown_value=-1))
])

Neighborhood_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('TE', TargetEncoder(smooth='auto', target_type='continuous')),
    ('scaler', StandardScaler()) # adding this since we use models that require scaling
])

transformers = [
    ('num', num_pipeline, num),
    ('cat', cat_pipeline, ['Bldg Type', 'House Style']),
    ('Kitchen', Kitchen_pipeline, ['Kitchen Qual']),
    ('Neighborhood', Neighborhood_pipeline, ['Neighborhood'])
]

processor = ColumnTransformer(transformers=transformers)

final_pipeline = Pipeline([
    ('processor', processor),
    ('model', XGBRegressor())
])

param_grid = [
    {
        'model' : [Ridge(random_state=42)],
        'model__alpha' : [0.1, 1.0, 10.0, 100.0, 500.0]
    },
    {
        'model' : [XGBRegressor(random_state=42)],
        'model__n_estimators' : [50,100,200,300],
        'model__learning_rate' : [0.01, 0.1, 0.2],
        'model__max_depth' : [3,5,7]
    },
    {
        'model' : [LinearSVR(random_state=42, max_iter=5000)],
        'model__C' : [10, 100, 1000, 10000],
        'model__epsilon' : [0, 1000, 5000] # error margin in dolars
    }
]

model = GridSearchCV(
    estimator = final_pipeline,
    param_grid = param_grid,
    cv = KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_absolute_error'
)

model.fit(X_train,y_train)

bs = -model.best_score_
bp = model.best_params_

print(bs)
print(bp)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test,y_pred)

print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape*100:.2f}%')

residuals = y_test - y_pred
plt.figure(figsize=(10,5))
plt.scatter(y_pred,residuals, alpha=0.5, c='magenta')
plt.show()

# Applying a log transformation to the target variable to stabilize variance and potentially improve model performance.

model_transformed = TransformedTargetRegressor(
    regressor = XGBRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.1),
    func=np.log1p,
    inverse_func=np.expm1
)

final_pipeline_transformed = Pipeline([
    ('processor', processor),
    ('model', model_transformed)
])

final_pipeline_transformed.fit(X_train, y_train)
y_pred_t = final_pipeline_transformed.predict(X_test)

mae_t = mean_absolute_error(y_test, y_pred_t)
mape_t = mean_absolute_percentage_error(y_test,y_pred_t)
print(f'MAE: {mae_t:.2f}')
print(f'MAPE: {mape_t*100:.2f}%')

residuals_t = y_test - y_pred_t
plt.figure(figsize=(10,5))
plt.scatter(y_pred_t,residuals_t, alpha=0.5, c='magenta')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(residuals_t, bins=30, kde=True)
plt.show()

names  = final_pipeline_transformed.named_steps['processor'].get_feature_names_out() # Used on Transformer
weights = final_pipeline_transformed.named_steps['model'].regressor_.feature_importances_

FI = pd.DataFrame({
    'Feature': names,
    'Importance': weights
})
FI = FI.sort_values(by=['Importance'], ascending=False)
print(FI)

FI.to_csv('Feature Importances.csv', index=False)

outputs = X_test.copy()
outputs['Real_Price'] = y_test
outputs['Predicted_Price'] = y_pred_t

outputs.to_csv('Business_outputs.csv', index=False)