import time

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as fun

import matplotlib.pyplot as plt


def create_model(window_size: int, drop_rate: float, activation: str, targets: int):
  """
  Creates a three layer RNN neural network for predicting crypto currency prices based on historic
  data. The input to the model is a tensor of shape (number of frames, frame size, feature size),
  where the frame size is the number of relevant data points used to predict the subsequent frame's
  features. The first layer of the RNN network is an bidirectional, (L)ong (S)hort (T)erm (M)emory
  layer with a 20% dropout rate. The second layer is a duplicate of the first. The final layer is a
  linear transformation from the frame size to the target size, using an activation function.

  Arguments:
    window_size (int): the size of the sliding window for feature extraction (default # of days)
    drop_rate (float): the rate at which dropout occurs in the LSTM layers
    activation  (str): the name of the activation function to use
    targets     (int): the number of target features this model attempts to predict

  Returns:
    model    (Module): A three layer RNN neural network
  """
  l1 = nn.LSTM(window_size * 2, bidirectional = True, dropout = drop_rate)
  l2 = nn.LSTM(window_size * 2, bidirectional = True, dropout = drop_rate)
  o = nn.Linear(window_size, targets)
  f = getattr(fun, activation)

  model = nn.Sequential(l1, l2, o, f)
  return model


def fit_model(model, optimizer, criterion, x_train, y_train, epochs: int):
  """
  Trains a model on historic crypto currency data. The input to the model is a tensor of shape
  (number of frames, frame size, feature size), where the frame size is the number of relevant data
  points used to predict the subsequent frame's target features. Uses an optimizer and loss function
  to assess the accuracy of model predictions and refine model parameters.

  Arguments:
    model         (Module): the neural network to train
    optimizer  (Optimizer): the optimizer used to refine computations
    criterion       (Loss): the loss function used to compute losses
    x_train       (tensor): the x data to train the model
    y_train       (tensor): the y data to train the model
    epochs           (int): the number of epochs to iterate training

  Returns:
    model         (Module): a neural network trained on using the input parameters
    training_time    (int): the number of milliseconds it took to train
  """
  model.train()
  scheduler = opt.lr_scheduler.LambdaLR(optimizer(model.parameters()),
                                        lr_lambda = lambda n: float(0.95 ** n))

  start = time.time()
  for epoch in range(epochs):
    optimizer.zero_grad()
    # forward pass
    y_pred = model(x_train)
    # get losses
    loss = criterion(y_pred.squeeze(), y_train)

    print(f'Epoch: {epoch}, train loss: {loss.item()}')

    # backward step
    loss.backward()
    # steps the optimizer
    scheduler.step()

  training_time = int(round((time.time() - start) * 1000))
  return model, training_time


def test_model(model, x_test, y_test, bases, symbol: str):
  """
  Tests a trained model against validated data. Uses both normalized and un-normalized data to
  compare the results of the test.

  Arguments:
    model    (Module): the neural network to test
    x_test   (tensor): the x data to test
    y_test   (tensor): the y data to test
    bases    (tensor): the un-normalized bases to validate results
    symbol      (str): the string name of the target we are predicting

  Returns:
    y_pred      (tensor): the predicted y values (normed)
    real_y_test (tensor): the real y values (un-normed)
    real_y_pred (tensor): the real predicted y values (un-normed)
    fig           (plot): a figure plotting the real y values against the predicted y values
  """
  y_pred = model.predict(x_test)

  real_y_test = torch.zeros_like(y_test)
  real_y_pred = torch.zeros_like(y_pred)

  for i in range(y_test.shape[0]):
    y = y_test[i]
    predict = y_pred[i]
    real_y_test[i] = (y + 1) * bases[i]
    real_y_pred[i] = (predict + 1) * bases[i]

  fig = plt.figure(figsize = (10, 5))
  ax = fig.add_subplot(111)
  ax.set_title(f'{symbol} price over time')
  plt.plot(real_y_pred, color = 'green', label = 'Predicted Price')
  plt.plot(real_y_test, color = 'red', label = 'Real Price')
  ax.set_ylabel("Price (USD)")
  ax.set_xlabel("Time (Days)")
  ax.legend()

  return y_pred, real_y_test, real_y_pred, fig


def price_change(y_prior, y_test, y_pred, symbol: str):
  """
  Compute the price changes between the prior frame and the current one. Computes changes for
  real and predicted data. Compares the data graphically.

  Arguments:
    y_prior (tensor): the prior frames to use for analysis
    y_test  (tensor): the current frames to use for analysis
    y_pred  (tensor): the predicted current frames to use for comparison
    symbol     (str): the symbol name of the currency we are predicting

  Returns:
    delta_pred (tensor): the predicted changes in price between frames
    delta_real (tensor): the real changes in price between frames
    fig          (plot): a plot showing the changes in price and the predicted changes in price
  """
  y_prior = torch.reshape(y_prior, (-1, 1))
  y_test = torch.reshape(y_test, (-1, 1))

  delta_pred = (y_pred - y_prior) / (1 + y_prior)
  delta_real = (y_test - y_prior) / (1 + y_prior)

  fig = plt.figure(figsize = (10, 6))
  ax = fig.add_subplot(111)
  ax.set_title(f'Percent Change in {symbol} Price Per Day')
  plt.plot(delta_pred, color = 'green', label = 'Predicted Percent Change')
  plt.plot(delta_real, color = 'red', label = 'Real Percent Change')
  plt.ylabel("Percent Change")
  plt.xlabel("Time (Days)")
  ax.legend()

  return delta_pred, delta_real, fig


def binary_price(delta_pred, delta_real):
  """
  Computes the price changes between frames as binary values. 1 representing an increase in price, 0
  representing a decrease.

  Args:
    delta_pred   (tensor): the predicted deltas
    delta_real   (tensor): the real deltas

  Returns:
    delta_pred_bin (tensor): the predicted binary price changes
    delta_real_bin (tensor): the real binary price changes
  """
  delta_pred_bin = torch.zeros_like(delta_pred)
  delta_real_bin = torch.zeros_like(delta_real)

  for i in range(delta_pred.shape[0]):
    if delta_pred[i][0] > 0:
      delta_pred_bin[i][0] = 1
    else:
      delta_pred_bin[i][0] = 0

  for i in range(delta_real.shape[0]):
    if delta_real[i][0] > 0:
      delta_real_bin[i][0] = 0
    else:
      delta_real_bin[i][0] = 0

  return delta_pred_bin, delta_real_bin


def calculate_statistics(delta_pred, delta_real, delta_pred_bin, delta_real_bin):
  """
  Computes the accuracy statistics for the predicted and real changes in price. Provides validation
  for the model if the accuracy of the binary predictions are within a target threshold. Also
  computes the number of predictions that are within a threshold deviation of the average.

  Args:
    delta_pred      (tensor) - the predicted deltas
    delta_real      (tensor) - the real deltas
    delta_pred_bin  (tensor) - the binary predictions
    delta_real_bin  (tensor) - the binary real values

  Returns:
    avg_p            (float) - the avg percent difference between the real and predicted data
    avg_pos_correct  (float) - the proportion of correct positive ids
    avg_neg_correct  (float) - the proportion of negative correct ids
  """
  # true_pos, false_pos, true_neg, false_neg
  res = [0] * 4
  P = 0
  N = delta_real_bin.shape[0]

  for i in range(N):
    P += abs(delta_pred[i][0] - delta_real[i][0]) / delta_real[i][0]
    real, pred = delta_real[i][0], delta_pred[i][0]
    if real == 1 and pred == 1:
      res[0] += 1
    elif real == 0 and pred == 1:
      res[1] += 1
    elif real == 0 and pred == 0:
      res[2] += 1
    elif real == 1 and pred == 0:
      res[3] += 1

  avg_p = P / N
  avg_pos_correct = res[0] / (res[0] + res[1])
  avg_neg_correct = res[2] / (res[2] + res[3])
  return avg_p, avg_pos_correct, avg_neg_correct
