import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def label(i):
    if i % 2 == 0:
        return 0
    else:
        return 1




random_numbers = np.random.randint(1, 1000, 4000)  # Puedes ajustar el rango y el tamaño según tus necesidades
X = random_numbers.reshape(-1, 1)
y = [label(n) for n in random_numbers]

x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(x_train, y_train)

predictions = model.predict(x_val)

mae = mean_absolute_error(y_val, predictions)

print('Measure abs error: ', mae)
print('Predicciones' ,model.predict(np.asarray([542, 764, 236, 650, 423, 657, 425, 549]).reshape(-1, 1)))
