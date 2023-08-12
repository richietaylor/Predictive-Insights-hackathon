import pandas as pd
import statsmodels.api as sm

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('train.csv')

# Define the independent variables
independent_vars = ['independent_var1', 'independent_var2', ..., 'independent_var15']

# Add a constant term to the independent variables matrix
X = sm.add_constant(df[independent_vars])

# Define the dependent variable
y = df['target']

# Create and fit the linear probability model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())
