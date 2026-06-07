LR, DT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv('fastfood.csv', na_values=['NA'])
display(data.head())
print(data.info())

data.isna().sum()

data_subset = data.loc[data['total_fat'] <= 125].copy()

print(f"Broj redova u data_subset: {len(data_subset)}")
print(f"Broj redova u originalnom datesetu: {len(data)}")
data_subset['protein'] = pd.to_numeric(data_subset['protein'], errors='coerce')
data_subset['cal_fat'] = pd.to_numeric(data_subset['cal_fat'], errors='coerce')

data_numeric = data_subset.select_dtypes(include=['number']).copy()

print(data_numeric.columns)
print(data_numeric.info())

data_numeric.isna().sum()
print((data_numeric.isna().mean() * 100).round(2))

cols_with_na = ['cal_fat', 'fiber', 'protein']

for col in cols_with_na:
    series = data_numeric[col].dropna()
    stat, p = shapiro(series)
    print(f"{col}: p = {p:.5f} → {'nije normalna' if p < 0.05 else 'normalna'} distribucija")

for col in cols_with_na:
    median = data_numeric[col].median()
    data_numeric[col] = data_numeric[col].fillna(median)

data_numeric.isna().sum()

corr = data_numeric.corr(numeric_only=True).round(4)
plt.figure(figsize=(11,8))
sb.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korelaciona matrica')
plt.show()

print(data_numeric[['total_fat', 'protein']].head(20))
target = 'protein'
lm1 = smf.ols(f'{target} ~ ' + ' + '.join(features), data=train_df).fit() #ovde smo ubacili train vrednosti umesto da pravimo x y train
print(lm1.summary())
features = ['cholesterol', 'calories', 'sodium']
train_df, test_df = train_test_split(data_subset, test_size=0.2, random_state=123)
y_true = test_df['protein']
X_test = test_df[['cholesterol', 'calories', 'sodium']]
y_pred = lm1.predict(X_test)
results = pd.DataFrame({
    'Stvarno': y_true,
    'Predviđeno': y_pred
})
print(results.head(10))

print(f"Broj redova u train skupu: {len(train_df)}")
print(f"Broj redova u test skupu: {len(test_df)}")

DT
sns.kdeplot(data=df, x='CreditScore', hue='Stayed')
plt.show()
df.drop(columns=['CreditScore'], inplace=True)

for col in ['Geography', 'Gender', 'Card.Type']:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, hue='Stayed', palette='pastel')
    plt.title(f'Distribucija {col} po klasi Stayed')
    plt.xlabel(col)
    plt.ylabel('Broj klijenata')
    plt.legend(title='Stayed')
    plt.tight_layout()
    plt.show()

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')
cats = ['Geography', 'Gender']
encoded_array = ohe.fit_transform(df[cats])
encoded_cols = ohe.get_feature_names_out(cats)

df_ohe = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
df = df.drop(columns=cats).join(df_ohe)

X = df.drop(columns='Stayed')
y = df['Stayed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold

cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {'ccp_alpha':np.arange(0.0,0.05,0.0025)}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy')
grid.fit(X_train, y_train)

best_ccp_alpha = grid.best_params_['ccp_alpha']
print("Najbolja vrednost ccp_alpha:",best_ccp_alpha)
tree = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha,random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=[30,10])
plot_tree(tree,
          feature_names=X_train.columns,
          class_names=['No','Yes'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Stablo odlučivanja - tree")
plt.show()
y_pred = tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Matrica kofuzije:")
print(cm)
rom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

knn_eval = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
print(knn_eval)

KNN
streams_3Q = df['streams'].quantile(0.75)
df['HighStreams'] = np.where(df['streams'] > streams_3Q, 'Yes', 'No')
df['HighStreams'] = df['HighStreams'].map({'Yes': 1, 'No': 0})
df.drop(columns='streams', inplace=True)
print(df.isna().sum())
p_spotify = shapiro(df['in_spotify_charts'].dropna())[1]
p_apple = shapiro(df['in_apple_charts'].dropna())[1]
df['in_spotify_charts'] = df['in_spotify_charts'].fillna(median_spotify)
df['in_apple_charts'] = df['in_apple_charts'].fillna(median_apple)
df.drop(columns='track_name', inplace=True)
sns.kdeplot(data=df, x='released_year', hue='HighStreams') #seaborn sb sns
plt.show()
numeric_vars = ['released_year', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts']
def count_outliers(variable):
    q1 = variable.quantile(0.25)
    q3 = variable.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((variable < lower) | (variable > upper)).sum()
outliers_per_column = df[numeric_vars].apply(count_outliers)
print("Broj autlajera po kolonama:\n", outliers_per_column)
shapiro_results = df[numeric_vars].apply(lambda x: shapiro(x)[1])
print("Shapiro-Wilk p-vrednosti:\n", shapiro_results)
robust_scaler = RobustScaler()
df_st = pd.DataFrame()
df_st[numeric_vars] = robust_scaler.fit_transform(df[numeric_vars])
print(df['mode'].unique())
df_st['mode'] = df['mode'].map({'Major': 0, 'Minor': 1})
df_st['HighStreams'] = df['HighStreams']
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(3, 26, 2))}
cv = GridSearchCV(estimator=knn,
                  param_grid=param_grid,
                  cv=10)
cv.fit(X_train, y_train)

best_k = cv.best_params_['n_neighbors']
print("Optimalna vrednost za k:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)
knn_eval = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
print(knn_eval)

Neur
from sklearn.neural_network import MLPClassifier
df = pd.read_csv("data/breast_cancer.csv", na_values=['-', ' ', ''])

display(df.head())
df.info()
X = df.drop(columns='isBenign')
y = df['isBenign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
mlp1=MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42, solver='sgd')
mlp1.fit(X_train, y_train)
plt.figure()
plt.plot(mlp1.loss_curve_)
plt.xlabel("Iteracija")
plt.ylabel("Loss")
plt.grid()
plt.show()
y_pred=mlp1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
knn_eval = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
print(knn_eval)
