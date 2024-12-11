import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.nn import MSELoss

from backprojector_fan import DifferentiableFanBeamBackprojector
from geometry import Geometry
from helper import params_2_proj_matrix, filter_ramlak, get_projection_matrices_2d


plt.ion()
device = torch.device('cuda')
n_iter = 100                   # 100ß
translation_amplitude = 5

def main():
    # geometry parameters for fan-beam CT
    detSourceDistance = 2000    # 2000
    detIsoDistance = 1000       # 1000
    n_frames =  300             # 300 / 400
    volume_pixels = 180         
    detector_pixels = 300

    # define scan geometry and create projection matrices
    geometry = Geometry((128, 128), (-63.5, -63.5), (1., 1.), -255., 2.)

    # volume_shape = (volume_pixels, volume_pixels)
    # volume_origin = (-((volume_pixels - 1) / 2), -((volume_pixels - 1) / 2))
    # volume_spacing = (1., 1.)
    # detector_origin = -(detector_pixels - 1)
    # detector_spacing = 2.
    # geometry = Geometry(volume_shape, volume_origin, volume_spacing, detector_origin, detector_spacing)
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    dsd = detSourceDistance * np.ones_like(angles)
    dsi = detIsoDistance * np.ones_like(angles)
    tx = np.zeros_like(angles)
    ty = np.zeros_like(angles)
    # true_proj_matrices  = get_projection_matrices_2d('./data/projection_matrices/P_CV_gt.txt', 400)
    true_proj_matrices, _, _ = params_2_proj_matrix(angles, dsd, dsi, tx, ty, geometry.detector_spacing, -geometry.detector_origin / geometry.detector_spacing)
    true_proj_matrices = torch.from_numpy(true_proj_matrices.astype(np.float32))
    true_proj_matrices = true_proj_matrices.to(device)

    # load sinogram
    sino = np.load('data/sinogram_filtered.npy')
    sino = torch.from_numpy(sino)
    sino = sino.to(device)

    # path = './data/imfusion_data'
    # sino = np.load(path + '/sino_2.npy')

    # normalize sinogram 
    # sino_mean = np.mean(sino)
    # sino_std = np.std(sino)
    # sino = (sino - sino_mean) / sino_std

    # apply ramlak filter to sinogram
    # sino = filter_ramlak(sino)
    
    # create 2x2 identity matrix as rotation for 400 projections
    rotation = torch.zeros((n_frames, 2, 2), device=device)
    rotation[:, 0, 0] = 1.
    rotation[:, 1, 1] = 1.

    # create random 2x1 translation vector for 400 projections and turn on gradient for translations
    translation = translation_amplitude * (torch.rand(n_frames, 2, 1, device=device) - 0.5)
    translation.requires_grad = True

    # create bottom row to complete 3x3 rigid motion matrix
    bottom_row = torch.zeros((n_frames, 1, 3), device=device)
    bottom_row[:, :, 2] = 1

    # instantiate differentiable backprojector for fan-beam geometry
    backprojector = DifferentiableFanBeamBackprojector.apply

    # compute initial motion-affected reconstruction and ground-truth motion-free reconstruction
    with torch.no_grad():
        motion = torch.cat((torch.cat((rotation, translation), dim=2), bottom_row), dim=1)
        perturbed_proj_matrices = torch.einsum('nij,njk->nik', true_proj_matrices, motion)
        # perturbed_proj_matrices  = get_projection_matrices_2d('data/projection_matrices/P_CV.txt', 400)
        # perturbed_proj_matrices = torch.from_numpy(perturbed_proj_matrices.astype(np.float32))
        # perturbed_proj_matrices = perturbed_proj_matrices.to(device)

        initial_reconstruction = backprojector(sino, perturbed_proj_matrices, geometry)
        target_reconstruction = backprojector(sino, true_proj_matrices, geometry)

    image, loss_plot, x, y = setup_plotting(initial_reconstruction)

    # setup optimizer and target function
    optimizer = SGD([translation], lr=1)
    # optimizer = SGD([translation], lr=5, momentum=0.9)
    loss_function = MSELoss()

    # run optimization of translation parameters
    for i in range(n_iter):
        optimizer.zero_grad()

        # assemble motion matrix and apply motion to projection matrices
        motion = torch.cat((torch.cat((rotation, translation), dim=2), bottom_row), dim=1)
        perturbed_proj_matrices = torch.einsum('nij,njk->nik', true_proj_matrices, motion)
        # perturbed_proj_matrices  = get_projection_matrices_2d('data/projection_matrices/P_CV.txt', 400)

        # run backprojection
        prediction = backprojector(sino, perturbed_proj_matrices, geometry)

        # compute loss
        loss = loss_function(prediction, target_reconstruction)

        # print('iteration: {}, loss: {}'.format(i, loss.item()))

        # compute gradient of loss wrt translation parameters
        loss.backward()

        # optimize translation parameters with SGD
        optimizer.step()

        # update plots
        image.set_data(prediction.detach().cpu().numpy())
        y[i] = loss.item()
        loss_plot[0].set_data(x, y)
        plt.draw()
        plt.pause(0.1)

    # compute motion-recovered reconstruction after optimization
    with torch.no_grad():
        motion = torch.cat((torch.cat((rotation, translation), dim=2), bottom_row), dim=1)
        perturbed_proj_matrices = torch.einsum('nij,njk->nik', true_proj_matrices, motion)
        recovered_reconstruction = backprojector(sino, perturbed_proj_matrices, geometry)


def setup_plotting(initial_reconstruction):
    x = list(range(n_iter))
    y = [None for i in range(n_iter)]
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3.4))
    image = ax[0].imshow(initial_reconstruction.cpu().numpy(), cmap='gray') # , vmin=0, vmax=20)
    ax[0].axis('off')
    ax[0].set_title('Reconstruction')
    loss_plot = ax[1].plot(x, y)
    ax[1].set_title('Loss')
    ax[1].set_xlim([0, n_iter])
    # ax[1].set_ylim([0, 0.1])
    ax[1].set_ylim([0, 20])
    ax[1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.draw()
    return image, loss_plot, x, y


if __name__ == '__main__':
    main()
