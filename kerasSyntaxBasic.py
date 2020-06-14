import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

# Main workflow step(1)-Preparing the data.
# Import the data csv file.
df = pd.read_csv("DATA/fake_reg.csv")
# print(df)

# Simply visualize the data correlation.
# sns.pairplot(df)
# plt.show()

# Convert the data to input data set and outptu data set.
# .values method convert the dataframe format to numpy array format.
# Since TF, Keras process data in numpy array format.
X = df[["feature1", "feature2"]].values
y = df["price"].values
# print(X.shape)
# print(y.shape)

# Split the data to train set and test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# print(X_train.shape)

# Normalize the input feature data on both train and test set.
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train)

# Main workflow step(2)-Creating and Training the ANN model.
# Create a simple ANN with 2 hidden layers by using Keras.
# All hidden layers are with 4 nodes with Relu activation function.
# Output layer with 1 node only since we are performing regression problem.
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1))

# Comply the ANN model by .comply method.
model.compile(optimizer="rmsprop", loss="mse")

# Train the ANN model by .fit method.
model.fit(x=X_train, y=y_train, epochs=250, verbose=1)

# Put the loss into a data frame object and plot it against number of epochs.
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

# Main workflow step(3)-Evalution of the trained ANN model.
model.evaluate(X_test, y_test, verbose=0)
model.evaluate(X_train, y_train, verbose=0)

# Perform prediction on the test set using the trained model.
test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=["Test True Y"])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ["Test True Y", "Model Predictions"]
# print(pred_df)
sns.scatterplot(x="Test True Y", y="Model Predictions", data=pred_df)
plt.show()

# metrics to compare the Actual y-test value vs. the predicted output value
mae = mean_absolute_error(pred_df["Test True Y"], pred_df["Model Predictions"])
mse = mean_squared_error(pred_df["Test True Y"], pred_df["Model Predictions"])
rmse = np.sqrt(mse)
# print(mae)
# print(mse)
# print(rmse)

new_test_data_point = [[998, 1000]]
new_test_data_point = scaler.transform(new_test_data_point)
print(model.predict(new_test_data_point))

# sexport and save the model in .h5 format for later use.
model.save("kerasSyntaxBasic.h5")

# open a previous saved model.
later_model = load_model("kerasSyntaxBasic.h5")
print(later_model.predict(new_test_data_point))
