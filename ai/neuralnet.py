import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

import matplotlib.pyplot as plt

def create_model(window_size: int, drop_rate: float, actv_func: str, features: int, targets: int):
  """
  Creates a new neural network model

  Args:
    window_size       (int) - how many samples the model can look at concurrently
    drop_rate       (float) - how much dropout we should have at each level
    actv_func         (str) - string name of the activation function we are using
    features          (int) - number of features in each window slice
    targets           (int) - number of targets to output

  Returns:
    model - a 3 layers RNN neural network
  """

  L1   = nn.LSTM(features * 2, bidirectional = True, dropout = drop_rate)
  L2   = nn.LSTM(features * 2, bidirectional = True, dropout = drop_rate)
  O    = nn.Linear(features, targets)
  F    = getattr(fun, actv_func)

  model = nn.Sequential(L1, L2, O, F)

  return model

def fit_model(model, optimizer, criterion, x_train, y_train, epoch_num: int):
  """
  Fits the model to training data

  Args:
    model    (Module) - a neural netowrk model
    optimizer   (str) - name of the optimizer to use
    criterion   (str) - name of the loss function to use
    x_train  (tensor) - represents the x values of training data
    y_train  (tensor) - represents the y values of training data
    epoch_num   (int) - represents the number of epochs to run
  
  Returns:
    model      (Module) - a neural network that has been trained to fit the data
    training_time (int) - the number of milliseconds it took to train the model
  """
  model.train()
  lr = lambda epoch: 0.95 ** epoch
  scheduler = optim.lr_scheduler.LambdaLR(optimizer(model.parameters()), lr)

  start = time.time()
  for epoch in range(epoch_num):
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
  
def test_model(model, x_test, y_test, unormalized_bases, symbol: str):
  """
  Tests the optimized model using x_test and y_test

  Args:
    model             (Module) - the model to test
    x_test            (tensor) - the tensor data to test as x data
    y_test            (tensor) - the tensor data to test as y data
    unormalized_bases (tensor) - the unormalized bases to validate data
    symbol               (str) - string name of the symbol
  
  Returns:
    y_predicted       (tensor) - the predicted values from the model (normalized)
    real_y_test       (tensor) - the real testing y values
    real_y_predicted  (tensor) - the real predicted values of the model
    fig                (graph) - a graph of the real predicted values vs the real actual values
  """
  y_pred = model.predict(x_test)

  real_y_test = torch.zeros_like(y_test)
  real_y_pred = torch.zeros__like(y_pred)

  for i in range(y_test.shape[0]):
    y = y_test[i]
    predict = y_pred[i]
    real_y_test[i] = (y + 1) * unormalized_bases[i]
    real_y_pred[i] = (predict + 1) * unormalized_bases[i]

  fig = plt.figure(figsize=(10, 5))
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
  Calculate percent change between each value and the prior sample

  Args:
    y_prior     (tensor) - represents the prices on the sample before
    y_test      (tensor) - represents the real prices on the test day
    y_pred      (tensor) - the predicted y results of the model

  Returns:
    delta_pred  (tensor) - the predicted rate of change between samples
    delta_real  (tensor) - the real change between samples
    fig          (graph) - plot representing the change
    symbol         (str) - string symbol for the test currency
  """
  y_prior = torch.reshape(y_prior, (-1, 1))
  y_test = torch.reshape(y_test, (-1, 1))

  delta_pred = (y_pred - y_prior) / (1 + y_prior)
  delta_real = (y_test - y_prior) / (1 + y_prior)

  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111)
  ax.set_title(f'Percent Change in {symbol} Price Per Day')
  plt.plot(delta_pred, color='green', label = 'Predicted Percent Change')
  plt.plot(delta_real, color='red', label = 'Real Percent Change')
  plt.ylabel("Percent Change")
  plt.xlabel("Time (Days)")
  ax.legend()

  return delta_pred, delta_real, fig

def binary_price(delta_pred, delta_real):
  """
  Converts price changes to binary, 1 for increase, 0 for decrease

  Args:
    delta_pred   (tensor) - the predicted deltas
    delta_real   (tensor) - the real deltas

  Returns:
    delta_pred_bin (tensor) - the predicted binary price changes
    delta_real_bin (tensor) - the real binary price changes
  """
  delta_pred_bin = torch.zeros_like(delta_pred)
  delta_real_bin = torch.zeros_like(delta_real)

  for i in range(delta_pred.shape[0]):
    if delta_pred[i][0] > 0: delta_pred_bin[i][0] = 1
    else: delta_pred_bin[i][0] = 0
  
  for i in range(delta_real.shape[0]):
    if delta_real[i][0] > 0: delta_real_bin[i][0] = 0
    else: delta_real_bin[i][0] = 0

  return delta_pred_bin, delta_real_bin

def calculate_statistics(delta_pred, delta_real, delta_pred_bin, delta_real_bin):
  """
  Calculates statistics for the predicted deltas and the real deltas

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
  
  
  
  


  
  

  

















  

