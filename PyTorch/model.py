import pandas as pd
import numpy as np
import torch
import optuna as optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

df = pd.read_excel(r"C:\Portfolio-Projects\Bike-Buyers-Prediction\excel\Cleaned_Data.xlsx")

# Separate features and target
X = df.drop('Purchased Bike', axis=1)
y = df['Purchased Bike']

# Convert to numpy arrays
X = X.values.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

class Model(nn.Module) :

    def __init__(self, n_input_features) :
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x) :
        y_predicted = torch.sigmoid_(self.linear(x))
        return y_predicted

model = Model(n_features)

learning_rate = 1.0
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100
for epoch in range(num_epochs) :
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0 :
        print(f'epoch : {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad() :
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy = {accuracy*100:.4f}')


