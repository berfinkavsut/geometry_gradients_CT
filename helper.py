from numba import cuda

import numpy as np
import cv2



@cuda.jit
def vox2world_cuda(voxel_index, spacing, origin):
    return voxel_index * spacing + origin


@cuda.jit
def world2vox_cuda(world_index, spacing, origin):
    return (world_index - origin) / spacing


@cuda.jit
def interpolate1d_cuda(array, pos):
    pos_floor = int(pos)
    pos_floor_plus_one = pos_floor + 1
    delta = pos - pos_floor

    # Check boundary conditions
    if pos_floor < 0 or pos_floor > len(array) - 1:
        pos_floor = None
    if pos_floor_plus_one < 0 or pos_floor_plus_one > len(array) - 1:
        pos_floor_plus_one = None

    # Get function values of the data points
    a = 0.0
    b = 0.0
    if pos_floor is not None:
        a = array[int(pos_floor)]
    if pos_floor_plus_one is not None:
        b = array[int(pos_floor_plus_one)]

    return delta * b + (1 - delta) * a


@cuda.jit
def interpolate2d_cuda(array, pos_x, pos_y):
    pos_x_floor = int(pos_x)
    pos_x_floor_plus_one = pos_x_floor + 1
    delta_x = pos_x - pos_x_floor
    pos_y_floor = int(pos_y)
    pos_y_floor_plus_one = pos_y_floor + 1
    delta_y = pos_y - pos_y_floor

    # Check boundary conditions
    if pos_x_floor < 0 or pos_x_floor > array.shape[0] - 1:
        pos_x_floor = None
    if pos_x_floor_plus_one < 0 or pos_x_floor_plus_one > array.shape[0] - 1:
        pos_x_floor_plus_one = None
    if pos_y_floor < 0 or pos_y_floor > array.shape[1] - 1:
        pos_y_floor = None
    if pos_y_floor_plus_one < 0 or pos_y_floor_plus_one > array.shape[1] - 1:
        pos_y_floor_plus_one = None

    # Get function values of the data points
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    if pos_x_floor is not None and pos_y_floor is not None:
        a = array[int(pos_x_floor), int(pos_y_floor)]
    if pos_x_floor_plus_one is not None and pos_y_floor is not None:
        b = array[int(pos_x_floor_plus_one), int(pos_y_floor)]
    if pos_x_floor is not None and pos_y_floor_plus_one is not None:
        c = array[int(pos_x_floor), int(pos_y_floor_plus_one)]
    if pos_x_floor_plus_one is not None and pos_y_floor_plus_one is not None:
        d = array[int(pos_x_floor_plus_one), int(pos_y_floor_plus_one)]

    tmp1 = delta_x * b + (1 - delta_x) * a
    tmp2 = delta_x * d + (1 - delta_x) * c

    return delta_y * tmp2 + (1 - delta_y) * tmp1


def params_2_proj_matrix(angles, dsd, dsi, tx, ty, det_spacing, det_origin):
    ''' compute fan beam projection matrices from parameters for circular trajectory

    :param angles: projection angles in radians
    :param dsd: source to detector distance
    :param dsi: source to isocenter distance
    :param tx: additional detector offset in x (usually 0 for motion free, ideal trajectory)
    :param ty: additional detector offset in y (usually 0 for motion free, ideal trajectory)
    :param det_spacing: detector pixel size
    :param det_origin: attention!! this is (-detector_origin / detector_spacing) or simply (image_size - 0.5)!!
    :return:
    '''
    num_angles = len(angles)
    matrices = np.zeros((num_angles, 2, 3))
    matrices[:, 0, 0] = -dsd * np.sin(angles) / det_spacing + det_origin * np.cos(angles)
    matrices[:, 0, 1] = dsd * np.cos(angles) / det_spacing + det_origin * np.sin(angles)
    matrices[:, 0, 2] = dsd * tx / det_spacing + det_origin * (dsi + ty)
    matrices[:, 1, 0] = np.cos(angles)
    matrices[:, 1, 1] = np.sin(angles)
    matrices[:, 1, 2] = dsi + ty

    intrinsics = np.zeros((num_angles, 2, 2))
    intrinsics[:, 0, 0] = dsd / det_spacing
    intrinsics[:, 0, 1] = det_origin
    intrinsics[:, 1, 1] = 1.

    extrinsics = np.zeros((num_angles, 2, 3))
    extrinsics[:, 0, 0] = - np.sin(angles)
    extrinsics[:, 0, 1] = np.cos(angles)
    extrinsics[:, 0, 2] = tx
    extrinsics[:, 1, 0] = np.cos(angles)
    extrinsics[:, 1, 1] = np.sin(angles)
    extrinsics[:, 1, 2] = dsi + ty

    assert np.allclose(matrices, np.einsum('aij,ajk->aik', intrinsics, extrinsics))

    # normalize w.r.t. lower right entry
    matrices = matrices / matrices[:, 1, 2][:, np.newaxis, np.newaxis]

    return matrices, extrinsics, intrinsics

def filter_ramlak(sinogram):
    """
    Filter a given sinogram using a ramp filter in Fourier space.
    """

    Nproj, Npix = np.shape(sinogram)

    # Generate basic ramp filter (hint: there is the function np.fft.fftfreq.
    # Try it and see what it does. Watch out for a possible fftshift)
    ramp_filter = np.abs(np.fft.fftfreq(Npix))

    # filter the sinogram in Fourier space in detector pixel direction
    # Use the np.fft.fft along the axis=1
    sino_ft = np.fft.fft(sinogram, axis=1)

    # Multiply the ramp filter onto the 1D-FT of the sinogram and transform it
    # back into spatial domain
    sino_filtered = np.fft.ifft(sino_ft*ramp_filter, axis=1).real

    return sino_filtered

def decompose_projection_matrix(P):
    # Decompose the projection matrix
    K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    
    # Convert t from homogeneous coordinates to 3x1 vector
    t = t[:3] / t[3]
    
    return K, R, t

def read_projection_matrix_from_file(file_path, frames):
    
    proj_matrices = [] 

    with open(file_path, 'r') as file:

        for i in range(frames):

            values = []
            for j in range(3):            
                line = file.readline().strip()
                
                # Convert the line to a list of floats
                values_ = list(map(float, line.split()))
                values.extend(values_)
            
            # Convert the list to a 3x4 matrix
            P = np.array(values).reshape(3, 4)
            proj_matrices.append(P)


    return np.asarray(proj_matrices)

def convert_3d_proj_to_2d_proj(K, R, t):
    K_2d = np.zeros(shape=(2,2))
    K_2d[0,0]= K[0,0]
    K_2d[0,1]= K[0,2]
    K_2d[1,0]= K[2,0]
    K_2d[1,1]= K[2,2]
    
    t_2d = t[:2]
    R_2d = np.eye(2)
    return K_2d, R_2d, t_2d

def get_projection_matrices_2d(path, frames):
    proj_matrices = read_projection_matrix_from_file(path, frames)
    print(proj_matrices.shape)
    proj_matrices_2d = [] 

    for i in range(400):     
        K, R, t = decompose_projection_matrix(proj_matrices[i, :, :])
        K_, R_, t_ = convert_3d_proj_to_2d_proj(K, R, t)
        P_ = K_ @ np.hstack((R_, t_.reshape(2,1)))
        proj_matrices_2d.append(P_)

    return np.array(proj_matrices_2d)