# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from torch.autograd import grad
from PIL import Image


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]  # same as batch size, where in row i there is noise using some sigma that will be added to data point in row i.
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)  # Shape: (batch_sz, channels, img_sz, img_sz)  e.g: (32, 1, 32, 32)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss, score

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss, score

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:

      # Fix the network parameters:
      for p in model.parameters():
        p.requires_grad = False

      # Ask for the process to calcutate gradients of the data:
      batch.requires_grad = True

      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss, score = loss_fn(model, batch)

      print('score size:', score.size())  # (32, 1, 32, 32)

      # ------------------------------- Compute dScore/dX using autograd.grad() ------------------------------------------

      # Description:
      # In dScore_dx, if we look on one data point (e.g. dScore_dx[0] which is 1024x1024): 
      #     then each row i is the gradient of pixel i in the score w.r.t each entry in the batch.

      score_flat = score.view(score.size(0), -1)  # (32, 1*32*32)
      dScore_dx = torch.zeros(score.size(0), score_flat.size(1), score_flat.size(1)).double()   # (32, 1*32*32, 1*32*32)
      H = torch.zeros(score.size(0), score_flat.size(1), score_flat.size(1)).double()   # (32, 1*32*32, 1*32*32)

      for i in range(batch.size(0)):
        for j in range(score_flat.size(1)):
          if batch.grad is not None:
            batch.grad.data.zero_()
          grads = grad(outputs=score_flat[i, j], inputs=batch, retain_graph=True)[0]  # (32, 1, 32, 32)
          grads_i = grads[i, :, :, :]  # (1, 32, 32), should be size of one data point (gets the grads of score_flat[i,j] w.r.t data point i only)
          dScore_dx[i, j, :] = torch.flatten(grads_i)

      print('\ndScore_dx size:', dScore_dx.size())  # (32, 1*32*32, 1*32*32)
      dScore_dx_np = dScore_dx.cpu().detach().numpy()

      # Save dScore/dX of one data point:
      dScore_dx_datapoint0 = dScore_dx_np[0]
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_datapoint0.npy', dScore_dx_datapoint0)
      I8 = (((dScore_dx_datapoint0 - dScore_dx_datapoint0.min()) / (dScore_dx_datapoint0.max() - dScore_dx_datapoint0.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_datapoint0.png')

      # Save Absolute(dScore/dX) of one data point:
      dScore_dx_datapoint0_abs = np.absolute(dScore_dx_np[0])
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_datapoint0_abs.npy', dScore_dx_datapoint0_abs)
      I8 = (((dScore_dx_datapoint0_abs - dScore_dx_datapoint0_abs.min()) / (dScore_dx_datapoint0_abs.max() - dScore_dx_datapoint0_abs.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_datapoint0_abs.png')

      # Save Mean(Absolute(dScore/dX)):
      dScore_dx_mean_abs = np.mean(np.absolute(dScore_dx_np), axis=0)
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_mean_abs.npy', dScore_dx_mean_abs)
      I8 = (((dScore_dx_mean_abs - dScore_dx_mean_abs.min()) / (dScore_dx_mean_abs.max() - dScore_dx_mean_abs.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_mean_abs.png')

      # Create H table:
      for x in range(dScore_dx.size(0)):
        for i in range(dScore_dx.size(1)):
          for j in range(dScore_dx.size(2)):
            H[x, i, j] = dScore_dx[x, i, j] - dScore_dx[x, j, i]
    
      print('\nH_size:', H.size())  # (32, 1*32*32, 1*32*32)
      H_np = H.cpu().detach().numpy()

      # Save H of one data point:
      H_datapoint0 = H_np[0]
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_datapoint0.npy', H_datapoint0)
      I8 = (((H_datapoint0 - H_datapoint0.min()) / (H_datapoint0.max() - H_datapoint0.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_datapoint0.png')

      # Save Absolute(H) of one data point:
      H_datapoint0_abs = np.absolute(H_np[0])
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_datapoint0_abs.npy', H_datapoint0_abs)
      I8 = (((H_datapoint0_abs - H_datapoint0_abs.min()) / (H_datapoint0_abs.max() - H_datapoint0_abs.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_datapoint0_abs.png')

      # Save Mean(H):
      H_mean = np.mean(H_np, axis=0)
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean.npy', H_mean)
      I8 = (((H_mean - H_mean.min()) / (H_mean.max() - H_mean.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean.png')

      # Save Mean(Absolute(H)):
      H_mean_abs = np.mean(np.absolute(H_np), axis=0)
      np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_abs.npy', H_mean_abs)
      I8 = (((H_mean_abs - H_mean_abs.min()) / (H_mean_abs.max() - H_mean_abs.min())) * 255.9).astype(np.uint8)
      img = Image.fromarray(I8)
      img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_abs.png')

      1/0
      # ---------------------------------------------------------------------------------------------------------------------



      # -------------------- Compute dScore/dX using backward() function and batch.grad.zero() ------------------------------
      
      # score_flat = score.view(score.size(0), -1)  # (32, 1*32*32)
      # dScore_dx = torch.zeros(score.size(0), score_flat.size(1), score_flat.size(1)).double()   # (32, 1*32*32, 1*32*32)
      # H = torch.zeros(score.size(0), score_flat.size(1), score_flat.size(1)).double()   # (32, 1*32*32, 1*32*32)

      # for i in range(batch.size(0)):
      #   for j in range(score_flat.size(1)):
      #     # optimizer.zero_grad()
      #     if batch.grad is not None:
      #       batch.grad.data.zero_()

      #     score_flat[i, j].backward(retain_graph=True)
      #     grads_i = batch.grad[i]  # (1, 32, 32), should be size of one data point (gets the grads of score_flat[i,j] w.r.t data point i only)
      #     dScore_dx[i, j, :] = torch.flatten(grads_i)

      # print('\ndScore_dx size:', dScore_dx.size()) # (32, 1*32*32, 1*32*32)

      # # dScore_dx_mean = torch.mean(dScore_dx, 0)
      # # dScore_dx_mean_np = dScore_dx_mean.cpu().detach().numpy()
      # dScore_dx0_np = dScore_dx[0].cpu().detach().numpy()
      # # np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_mean_w_zerograd.npy', dScore_dx_mean_np)
      # np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/check_zero_grad/dScore_dx0_backward_w_gradzero.npy', dScore_dx0_np)

      # # I8 = (((dScore_dx_mean_np - dScore_dx_mean_np.min()) / (dScore_dx_mean_np.max() - dScore_dx_mean_np.min())) * 255.9).astype(np.uint8)
      # # img = Image.fromarray(I8)
      # # img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/dScore_dx_mean_w_zerograd.png')

      # I8 = (((dScore_dx0_np - dScore_dx0_np.min()) / (dScore_dx0_np.max() - dScore_dx0_np.min())) * 255.9).astype(np.uint8)
      # img = Image.fromarray(I8)
      # img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/check_zero_grad/dScore_dx0_backward_w_gradzero.png')

      # # Create H table:
      # for x in range(dScore_dx.size(0)):
      #   for i in range(dScore_dx.size(1)):
      #     for j in range(dScore_dx.size(2)):
      #       H[x, i, j] = dScore_dx[x, i, j] - dScore_dx[x, j, i]
      
      # H_mean = torch.mean(H, 0)  # (1*32*32, 1*32*32)
      # print('H_mean size:', H_mean.size())

      # H_mean_np = H_mean.cpu().detach().numpy()
      # np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_backward_w_zerograd2.npy', H_mean_np)
      # H0_np = H[0].cpu().detach().numpy()
      # np.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H0_backward_w_zerograd2.npy', H0_np)

      # # Normalize the data to be between [0,1] and then multiply by 256 so they will be between [0, 256].
      # # 0 is black and 256 is white. So low values will be darker than high values.
      # I8 = (((H_mean_np - H_mean_np.min()) / (H_mean_np.max() - H_mean_np.min())) * 255.9).astype(np.uint8)
      # img = Image.fromarray(I8)
      # img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_backward_w_zerograd2.png')

      # I8 = (((H0_np - H0_np.min()) / (H0_np.max() - H0_np.min())) * 255.9).astype(np.uint8)
      # img = Image.fromarray(I8)
      # img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H0_backward_w_zerograd.png')

      # ---------------------------------------------------------------------------------------------------------------------
      

      #loss.backward()

      #optimize_fn(optimizer, model.parameters(), step=state['step'])
      #state['step'] += 1
      #state['ema'].update(model.parameters())

    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss, score = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
