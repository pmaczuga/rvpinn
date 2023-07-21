import copy
from typing import Union
import numpy as np
import torch
from src.nn_error import NNError

from src.pinn import PINN
from src.loss.loss import Loss
from src.utils import TrainResult


def train_model(
    pinn: PINN,
    loss_fn: Loss,
    learning_rate: float = 0.01,
    max_epochs: int = 1_000,
    atol: float = 0.0001,
    rtol: float = 0.0001,
    best: bool = False,
    nn_error: Union[NNError, None] = None
) -> TrainResult:

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)
    loss_vector = torch.zeros(max_epochs)
    if nn_error:
       error_vector = torch.zeros(max_epochs)
       norm_vector = torch.zeros(max_epochs)
    if best == True:
      best_loss = torch.tensor(1e30)
      best_epoch = 0
      counter = -1
    
    for epoch in range(max_epochs):

        try:
          if nn_error:
            error_vector[epoch] = nn_error.error(pinn)
            norm_vector[epoch] = nn_error.norm(pinn)
          loss: torch.Tensor = loss_fn(pinn)
          loss_vector[epoch] = float(loss)
          optimizer.zero_grad()
          loss.backward(retain_graph=True)
          optimizer.step()
  

          if epoch == 0:
            loss0 = float(loss)
            print("-------------------------------------------------------------------------------------------------")
            print(f"Epoch: {epoch:>05d} - Loss: {float(loss):>.15f} - Relative Loss: {float(loss)/loss0:>.15f}")
            print("-------------------------------------------------------------------------------------------------")
  

          if (epoch+1) % 500 == 0:
              print(f"Epoch: {epoch:>05d} - Loss: {float(loss):>.15f} - Relative Loss: {float(loss)/loss0:>.15f}")
              if best == True:
                print(f"Best Epoch: {best_epoch:>05d} - Best Loss: {float(best_loss):>.15f} - Relative Best Loss: {float(best_loss)/loss0:>.15f}")
                print("-------------------------------------------------------------------------------------------------")


          if float(loss)<atol or float(loss)/loss0<rtol:
            print(f"Epoch: {epoch:>05d} - Loss: {float(loss):>.15f} - Relative Loss: {float(loss)/loss0:>.15f}")
            break

        except KeyboardInterrupt:
            break

        if best == True:
              if float(loss)< best_loss:
                state_dict = copy.deepcopy(pinn.state_dict())
                best_loss: torch.Tensor = loss_fn(pinn)
                best_epoch = epoch
                counter += 1
                #saving the state of the neural network to state.pth file that is kept on colab and removed when we close colab
                #torch.save(pinn.state_dict(), "delta_weak_%s.pth"%counter)
                #download the state.pth file to your local disc 
                #files.download("delta_weak_%s.pth"%counter) 

    if best == True:
        pinn.load_state_dict(state_dict)
        final_loss : torch.Tensor = loss_fn(pinn)
        print(f" Best Epoch: {best_epoch:>05d} - Best Loss: {final_loss:>.15f} - Relative Best Loss: {final_loss/loss0:>.15f}  ")
    
    if nn_error:
       return TrainResult(loss_vector, error_vector, norm_vector)
    return TrainResult(loss_vector)
