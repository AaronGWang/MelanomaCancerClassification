import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
import requests
import os
import tqdm

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents.

    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Print the total training time of a model
def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# A training loop for Multiclass models
def multiclass_train_loop(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               train_loss_list: list,
               train_acc_list: list):
    """Trains a model for one epoch on a set of data.

    Args:
        model (torch.nn.Module): a PyTorch model instance.
        train_dataloader (torch.utils.data.DataLoader): a PyTorch DataLoader instance.
        loss_fn (torch.nn.Module): a PyTorch loss function instance.
        optimizer (torch.optim.Optimizer): a PyTorch optimizer instance.
        accuracy_fn (function): a function to compute accuracy.
        train_loss_list (list): list to store training loss values.
        train_acc_list (list): list to store training accuracy values.

        
    Returns:
        None
    """
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(train_dataloader):

      y_logit = model(X)

      loss = loss_fn(y_logit, y)
      train_loss += loss
      train_acc += accuracy_fn(y_true=y,
                              y_pred=y_logit.argmax(dim=1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")


# A validation loop for Multiclass models
def multiclass_validation_loop(model: torch.nn.Module,
              validation_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              val_loss_list: list,
              val_acc_list: list):
  """Validates a model on a set of data.

  Args:
      model (torch.nn.Module): a PyTorch model instance.
      validation_dataloader (torch.utils.data.DataLoader): a PyTorch DataLoader instance.
      loss_fn (torch.nn.Module): a PyTorch loss function instance.
      accuracy_fn (function): a function to compute accuracy.
      test_loss_list (list): list to store testing loss values.
      test_acc_list (list): list to store testing accuracy values.

  Returns:
      None
  """

  test_loss, test_acc = 0, 0

  model.eval()

  with torch.inference_mode():
    for X, y in validation_dataloader:

      test_logit = model(X)

      test_loss += loss_fn(test_logit, y)
      test_acc += accuracy_fn(y_true=y, y_pred=test_logit.argmax(dim=1))

    test_loss /= len(validation_dataloader)
    test_acc /= len(validation_dataloader)

  val_loss_list.append(test_loss)
  val_acc_list.append(test_acc)
  print(f"Train Loss: {test_loss:.5f} | Train Acc: {test_acc:.2f}%\n")


# Evaluate the model based testing data
def eval_model(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
  """Evaluates a model on a set of data.
  Args:
      model (torch.nn.Module): a PyTorch model instance.
      test_dataloader (torch.utils.data.DataLoader): a PyTorch DataLoader instance.
      loss_fn (torch.nn.Module): a PyTorch loss function instance.
      accuracy_fn (function): a function to compute accuracy.

  Returns:
      dict: a dictionary of the model's loss and accuracy on the test data.
  """
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(test_dataloader):

      logit = model(X)

      loss += loss_fn(logit, y)
      acc += accuracy_fn(y_true=y, y_pred=logit.argmax(dim=1))

    loss /= len(test_dataloader)
    acc /= len(test_dataloader)

  return {"Model Name": model.__class__.__name__,
          "Model Test Loss": loss.item(),
          "Model Test Acc": acc}