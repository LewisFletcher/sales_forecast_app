import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('kaggle_sales_data.csv')

# Columns: country, order_value_EUR, cost, date, category, customer_name, sales_manager, sales_rep, device_type, order_id

df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

country_categories = df['country'].astype('category')
category_categories = df['category'].astype('category')
device_categories = df['device_type'].astype('category')

df['country_encoded'] = country_categories.cat.codes
df['category_encoded'] = category_categories.cat.codes
df['device_encoded'] = device_categories.cat.codes
df['order_value_EUR'] = df['order_value_EUR'].replace({',': ''}, regex=True).astype(float)
df['day_of_week'] = df['date'].dt.dayofweek

# Mapping for categorical variables on output

mappings = {
    'country': dict(enumerate(country_categories.cat.categories)),
    'category': dict(enumerate(category_categories.cat.categories)),
    'device_type': dict(enumerate(device_categories.cat.categories))
}

reverse_mappings = {
    'country': {v: k for k, v in mappings['country'].items()},
    'category': {v: k for k, v in mappings['category'].items()},
    'device_type': {v: k for k, v in mappings['device_type'].items()}
}

available_countries = df['country'].unique()
available_categories = df['category'].unique()
available_devices = df['device_type'].unique()

# Save available options for API validation
options_info = {
    'countries': available_countries.tolist(),
    'categories': available_categories.tolist(),
    'device_types': available_devices.tolist()
}

with open('options_info.pkl', 'wb') as f:
    joblib.dump(options_info, f)

features = df[['day_of_year', 'month', 'year', 'country_encoded', 'category_encoded', 'device_encoded', 'day_of_week']]
target = np.log1p(df['order_value_EUR'])

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,          # prevent overly deep trees
    min_samples_leaf=10,   # each leaf needs at least 10 samples
    max_features='sqrt',   # each split only considers a subset of features
    random_state=42,
    n_jobs=-1
)
model.fit(features_train, target_train)

train_score = model.score(features_train, target_train)
test_score = model.score(features_test, target_test)

print('Model trained successfully. R^2 Scores:')
print(f'Training R^2 Score: {train_score:.4f}')
print(f'Testing R^2 Score: {test_score:.4f}')

sample_value = features_test.iloc[0:1]
predicted_value = np.expm1(model.predict(sample_value)[0])
actual_value = np.expm1(target_test.iloc[0])

print(f'\nSample Prediction:')
print(f'Predicted Order Value (EUR): {predicted_value:.2f}')
print(f'Actual Order Value (EUR): {actual_value:.2f}')

joblib.dump(model, 'sales_model.pkl')

model_info = {
    'features': ['day_of_year', 'month', 'year', 'country_encoded', 'category_encoded', 'device_encoded', 'day_of_week'],
    'mappings': mappings,
    'reverse_mappings': reverse_mappings,
    'target': 'order_value_EUR',
}

with open('model_info.pkl', 'wb') as f:
    joblib.dump(model_info, f)