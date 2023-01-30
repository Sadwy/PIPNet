from sklearn.neural_network import MLPRegressor
import pandas as pd

df_data = pd.read_csv('../caliberate4/mapping_50cm.csv')
X = df_data.drop(columns=['Mi_x', 'Mi_y'])
y = df_data.loc[:, ['Mi_x', 'Mi_y']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25, stratify=y,
                                                    random_state=2022)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

model_mlp = MLPRegressor()
