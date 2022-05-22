import torch
from torch.autograd import grad
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# # Create some dummy data.
# x = torch.ones(2, 2, requires_grad=True)
# gt = torch.ones_like(x) * 16 - 0.5  # "ground-truths" 

# # We will use MSELoss as an example.
# loss_fn = nn.MSELoss()

# # Do some computations.
# v = x + 2
# y = v ** 2

# # dy_dx = grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
# # print(f'dy/dx:\n {dy_dx}')

# x_flat = torch.flatten(x)
# y_flat = torch.flatten(y)
# dy_dx = torch.ones(x_flat.size(0), x_flat.size(0)).double()
# for i in range(x_flat.size(0)):
#     dy_dx[i, :] = torch.flatten(grad(outputs=y_flat[i], inputs=x, retain_graph=True)[0])

# print(dy_dx)

# # dv_dx = grad(outputs=v, inputs=x, grad_outputs=torch.ones_like(v))
# # print(f'dv/dx:\n {dv_dx}')


# H_mean = np.load('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_backward_w_zerograd.npy')
# H_mean_abs = np.absolute(H_mean)

arr = np.load('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/H_mean_abs.npy')

print('min value: ', arr.min())
print('max value: ', arr.max())
print('mean value: ', arr.mean())
print('\n')

print('values where i=j:')
print(arr[10,10])
print(arr[100,100])
print(arr[1000,1000])
# for i in range(arr.shape[0]):
#     if arr[i, i] != 0:
#         print('i: ', i)

print('\n')

print(arr[0])

print('\n')

print(arr[110])

# I8 = (((arr - arr.min()) / (arr.max() - arr.min())) * 255.9).astype(np.uint8)
# img = Image.fromarray(I8)
# img.save('/vilsrv-storage/tohamy/BNP/SDE/score_validation/output_grads/arr.png')


# i = 10
# for j in range(80):
#     print('i=', i, 'j=', j, 'val:', arr[i, j])

# print('\n')

#print(arr[100,0])
#print(arr[0,100])