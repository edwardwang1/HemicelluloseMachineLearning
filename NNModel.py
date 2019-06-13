import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import time
from keras.models import load_model

performTuning = True

start = time.time()
data = pd.read_csv("data.csv")

# Prepping Data

data = data.sample(frac=1).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P']
X = data[XLabels]

# Scaling X
sc = StandardScaler()
X = sc.fit_transform(X)


numData = len(data.index)
numTrain = int(numData * 0.7)
numTest = int(numData * .15)
# print(numTest, numTrain)
train_Frame, valid_Frame, test_Frame, train_valid_Frame = data.iloc[:numTrain, :], data.iloc[numTrain:-numTest,
                                                                                   :], data.iloc[-numTest:,
                                                                                       :], data.iloc[:-numTest:, :]


y_train, y_valid, y_test, y_train_valid = train_Frame['Yield'], valid_Frame['Yield'], test_Frame['Yield'], \
                                          train_valid_Frame['Yield']

X_train, X_valid, X_test, X_train_valid = X[:numTrain, :], X[numTrain:-numTest, :], X[-numTest:, :], X[:-numTest, :]

## Artificial Neural Network
print("Starting Neural Network------------------")
learningRates = [0.002, 0.005, 0.01, 0.02]
batchSizes = [64, 128, 256, 512, 1024]
dropoutRates = [0.00, 0.001, 0.01, 0.1]
errors = []

if performTuning:
    for lr_ in learningRates:
        for bs in batchSizes:
            for dr in dropoutRates:
                model = Sequential()
                model.add(Dense(units=12, activation='sigmoid', input_dim=12))
                model.add(Dropout(dr))
                model.add(Dense(units=12, activation='sigmoid'))
                model.add(Dense(units=6, activation='sigmoid'))
                model.add(Dense(units=6, activation='sigmoid'))
                model.add(Dense(units=1, activation='softplus'))

                sgd = SGD(lr=lr_)
                model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

                model.fit(X_train, y_train, epochs=3000, batch_size=bs, verbose=0)
                y_pred = model.predict(X_valid, batch_size=bs)
                y_pred = y_pred.flatten()

                error = metrics.mean_absolute_error(y_valid, y_pred)
                errors.append(error)

    index_of_lowest_error = np.argmin(errors)

    best_lr = learningRates[int(index_of_lowest_error / (len(batchSizes) * len(dropoutRates)))]  # Good
    best_bs = batchSizes[int((index_of_lowest_error % (len(batchSizes) * len(dropoutRates))) / len(dropoutRates))]  # Good
    best_dr = dropoutRates[index_of_lowest_error % len(dropoutRates)]  # Good
    print("ANN Best Learning Rate is: ", best_lr)
    print("ANN Best Batch Size is: ", best_bs)
    print("ANN Best Dropout Rate is: ", best_dr)
else:
    best_lr = 0.01
    best_bs = 128
    best_dr = 0.1

model = Sequential()
model.add(Dense(units=12, activation='sigmoid', input_dim=12))
model.add(Dropout(best_dr))
model.add(Dense(units=12, activation='sigmoid'))
model.add(Dense(units=6, activation='sigmoid'))
model.add(Dense(units=6, activation='sigmoid'))
model.add(Dense(units=1, activation='softplus'))

sgd = SGD(lr=best_lr)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train_valid, y_train_valid, epochs=3000, batch_size=best_bs, verbose=0)

# loss_and_metrics = model.evaluate(X_test, y_test,batch_size=best_bs)

# np.savetxt("train_valid.csv", X_train_valid, delimiter=',')
# np.savetxt("y.csv", y_train_valid)

y_pred = model.predict(X_test, batch_size=best_bs)
y_pred = y_pred.flatten()

model.save('NNModel.h5')

output = pd.DataFrame({'pred':y_pred, 'test': y_test})
output.to_csv("NNModelOutput.csv")

end = time.time()
duration = end - start
print("Execution Time is:", duration /60, "min")

print('ANN Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('ANN Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('ANN Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("Ending Neural Network Regression ------------------")


