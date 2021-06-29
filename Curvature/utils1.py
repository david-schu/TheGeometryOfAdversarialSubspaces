import torch
import numpy as np
import scipy
import skimage.draw

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib as mpl

def add_arrow(ax, vect, xrange, yx_offset=[1,1], linestyle='-', label='', text_color='k'):
    arrow_width = 0.0
    arrow_linewidth = 1
    arrow_headsize = 0.15
    arrow_head_length = 0.15
    arrow_head_width = 0.15
    vect_x = vect[0].item()
    vect_y = vect[1].item()
    ax.arrow(0, 0, vect_x, vect_y,
        width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length,
        fc='k', ec='k', linestyle=linestyle, linewidth=arrow_linewidth, length_includes_head=True)
    tenth_range_shift = xrange/10 # For shifting labels
    text_handle = ax.text(
        vect_x+(tenth_range_shift*yx_offset[1]),
        vect_y+(tenth_range_shift*yx_offset[0]),
        label,
        weight='bold',
        color=text_color,
        horizontalalignment='center',
        verticalalignment='center')


def plot_contours(ax, activity, contours, fits, yx_pts, yx_range, proj_vects=None, num_levels=10, plot_fits=False, title=''):
    """
    TODO: Remove plot_fits and allow contours & fits params to be None to avoid plotting them
    """
    vmin = np.min(activity)
    vmax = np.max(activity)
    cmap = plt.get_cmap('cividis')
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    # Plot contours
    x_mesh, y_mesh = np.meshgrid(*yx_pts[::-1])
    levels = np.linspace(vmin, vmax, num_levels)
    contsf = ax.contourf(x_mesh, y_mesh, activity,
        levels=levels, vmin=vmin, vmax=vmax, alpha=1.0, antialiased=True, cmap=cmap)
    if proj_vects is not None:
        # Add arrows
        proj_target = proj_vects[0]
        xrange = max(yx_range[1]) - min(yx_range[1])
        add_arrow(ax, proj_target, xrange, linestyle='-')
        proj_comparison = proj_vects[1]
        add_arrow(ax, proj_comparison, xrange, linestyle='--')
        proj_orth = proj_vects[2]
        add_arrow(ax, proj_orth, xrange, linestyle='-')
    # Add axis grid
    ax.set_aspect('equal')
    ax.plot(yx_range[1], [0,0], color='k', linewidth=0.5)
    ax.plot([0,0], yx_range[0], color='k', linewidth=0.5)
    if plot_fits:
        ax.scatter(contours[0], contours[1], s=4, color='r')
        ax.scatter(fits[0], fits[1], s=3, marker='*', color='k')
    ax.format(
        ylim=[np.min(yx_pts[0]), np.max(yx_pts[0])],
        xlim=[np.min(yx_pts[1]), np.max(yx_pts[1])],
        title=title
    )
    return contsf


def overlay_image(ax, images, y_pos, x_pos, offset, num_images_per_edge, vmin=None, vmax=None):
    image_y_pos = int((y_pos + 2) / 4 * num_images_per_edge) - 1
    image_x_pos = int((x_pos + 2) / 4 * num_images_per_edge) - 1
    height_ratio = images.shape[3] / np.maximum(*images.shape[3:])
    width_ratio = images.shape[4] / np.maximum(*images.shape[3:])
    inset_height = 0.35 * height_ratio
    inset_width = 0.35 * width_ratio
    arr_img = np.squeeze(images[image_y_pos, image_x_pos])
    imagebox = OffsetImage(arr_img, zoom=0.75, cmap='Gray')
    img = imagebox.get_children()[0]; img.set_clim(vmin=vmin, vmax=vmax)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, 
        xy=[x_pos, y_pos],
        xybox=offset,
        boxcoords='offset points',
        pad=0.0,
        arrowprops=dict(
            linestyle='--',
            arrowstyle='->',
            color='r'
    ))
    ax.add_artist(ab)
    return arr_img

    
def plot_ellipse(axis, center, shape, angle, color_val="auto", alpha=1.0):
    """
    Add an ellipse to given axis
    Parameters:
        axis [matplotlib.axes._subplots.AxesSubplot] axis on which ellipse should be drawn
        center [tuple or list] specifying [y, x] center coordinates
        shape [tuple or list] specifying [width, height] shape of ellipse
        angle [float] specifying angle of ellipse
        color_val [matplotlib color spec] specifying the color of the edge & face of the ellipse
        alpha [float] specifying the transparency of the ellipse
    Outputs:
        ellipse [matplotlib.patches.ellipse] ellipse object
    """
    y_cen, x_cen = center
    width, height = shape
    ellipse = matplotlib.patches.Ellipse(xy=[x_cen, y_cen], width=width,
        height=height, angle=angle, edgecolor=color_val, facecolor='none',
        alpha=alpha, fill=True)
    axis.add_artist(ellipse)
    ellipse.set_clip_box(axis.bbox)
    return ellipse

def get_single_unit_activations_torch(model, images, neuron, kwargs=None):
    """
    Parameters:
        model [pytorch nn.module or subclass]
        images [pytorch tensor] of shape [num_images, channels, height, width]
        neuron [int] indicating which neuron we are returning activations for
        kwargs

    Outputs:
        activations [pytorch tensor] feature activations for the given neuron and for all inputs
    """
    with torch.no_grad():
        activations = model(images)[:, neuron]
    return torch.squeeze(activations)


def get_single_unit_activations(model, images, neuron, kwargs=None):
    """
    Parameters:
        model [pytorch nn.module or subclass]
        images [np.ndarray] of shape [num_images, channels, height, width]
        neuron [int] indicating which neuron we are returning activations for
        kwargs

    Outputs:
        activations [np.ndarray] feature activations for the given neuron and for all inputs
    """
    images = torch.from_numpy(images)#.cuda()
    with torch.no_grad():
        activations = model(images)[:, neuron]
    return activations.cpu().numpy()


def get_all_activations(model, images, kwargs=None):
    """
    Parameters:
        model [pytorch nn.module or subclass]
        images [np.ndarray] of shape [num_images, channels, height, width]
        kwargs

    Outputs:
        activations [np.ndarray] feature activations for the given neuron and for all inputs
    """
    images = torch.from_numpy(images)#.cuda()
    with torch.no_grad():
        activations = model(images)
    return activations.cpu().numpy()



def hilbert_amplitude(data, padding=None):
    """
    Compute Hilbert amplitude envelope of data matrix
    Inputs:
        data: [np.ndarray] of shape [num_data, num_rows, num_cols]
        padding: [list of int] specifying how much 0-padding to use for FFT along the row & column dimension, respectively
            default is the closest power of 2 of maximum(num_rows, num_cols)
    Outputs:
        envelope: [np.ndarray] same shape as data, contains Hilbert envelope
    TODO: Bug when num_data = 1
    """
    cart2pol = lambda x,y: (np.arctan2(y,x), np.hypot(x, y))
    num_data, num_rows, num_cols = data.shape
    if padding is None or max(padding) <= largest_edge_size:
        # Amount of zero padding for fft2 (closest power of 2)
        Ny = np.int(2**(np.ceil(np.log2(num_rows))))
        Nx = np.int(2**(np.ceil(np.log2(num_cols))))
    else:
        Ny = np.int(padding[0])
        Nx = np.int(padding[1])
    # Analytic signal envelope for data
    # (Hilbet transform of each image)
    envelope = np.zeros((num_data, num_rows, num_cols), dtype=complex)
    # Fourier transform of data
    f_data = np.zeros((num_data, Ny, Nx), dtype=complex)
    # Filtered Fourier transform of data
    # Hilbert filters
    hil_filt = np.zeros((num_data, Ny, Nx))
    # Grid for creating filter
    freqsx = (2 / Nx) * np.pi * np.arange(-Nx / 2.0, Nx / 2.0)
    freqsy = (2 / Ny) * np.pi * np.arange(-Ny / 2.0, Ny / 2.0)
    (mesh_fx, mesh_fy) = np.meshgrid(freqsx, freqsy)
    (theta, r) = cart2pol(mesh_fx, mesh_fy)
    for data_idx in range(num_data):
        # Grab single datapoint
        datapoint = (data - np.mean(data, axis=0, keepdims=True))[data_idx, ...]
        # Convert datapoint into DC-centered Fourier domain
        f_datapoint = np.fft.fftshift(np.fft.fft2(datapoint, [Ny, Nx]))
        f_data[data_idx, ...] = f_datapoint
        # Find indices of the peak amplitude
        max_ys = np.abs(f_datapoint).argmax(axis=0) # Returns row index for each col
        max_x = np.argmax(np.abs(f_datapoint).max(axis=0))
        # Convert peak amplitude location into angle in freq domain
        fx_ang = freqsx[max_x]
        fy_ang = freqsy[max_ys[max_x]]
        theta_max = np.arctan2(fy_ang, fx_ang)
        # Define the half-plane with respect to the maximum
        ang_diff = np.abs(theta - theta_max)
        idx = (ang_diff > np.pi).nonzero()
        ang_diff[idx] = 2.0 * np.pi - ang_diff[idx]
        hil_filt[data_idx, ...] = (ang_diff < np.pi / 2.).astype(int)
        # Create analytic signal from the inverse FT of the half-plane filtered datapoint
        abf = np.fft.ifft2(np.fft.fftshift(hil_filt[data_idx, ...] * f_datapoint))
        envelope[data_idx, ...] = abf[0:num_rows, 0:num_cols]
    return envelope


def gaussian_fit(pyx):
    """
    Compute the expected mean & covariance matrix for a 2-D gaussian fit of input distribution
    Inputs:
        pyx: [np.ndarray] of shape [num_rows, num_cols] that indicates the probability function to fit
    Outputs:
        mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    """
    assert pyx.ndim == 2, (
        'Input must have 2 dimensions specifying [num_rows, num_cols]')
    mean = np.zeros((1, 2), dtype=np.float32) # [mu_y, mu_x]
    for idx in np.ndindex(pyx.shape): # [y, x] ticks columns (x) first, then rows (y)
        mean += np.asarray([pyx[idx]*idx[0], pyx[idx]*idx[1]])[None, :]
    cov = np.zeros((2,2), dtype=np.float32)
    for idx in np.ndindex(pyx.shape): # ticks columns first, then rows
        cov += np.dot((idx-mean).T, (idx-mean))*pyx[idx] # typically an outer-product
    return (np.squeeze(mean), cov)


def generate_gaussian(shape, mean, cov):
    """
    Generate a Gaussian PDF from given mean & cov
    Inputs:
        shape: [tuple] specifying (num_rows, num_cols)
        mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    Outputs:
        tuple containing (Gaussian PDF, grid_points used to generate PDF)
            grid_points are specified as a tuple of (y,x) points
    """
    y_size, x_size = shape
    y = np.linspace(0, y_size, np.int32(np.floor(y_size)))
    x = np.linspace(0, x_size, np.int32(np.floor(x_size)))
    mesh_x, mesh_y = np.meshgrid(x, y)
    pos = np.empty(mesh_x.shape + (2,)) #x.shape == y.shape
    pos[:, :, 0] = mesh_y; pos[:, :, 1] = mesh_x
    gauss = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    return (gauss.pdf(pos), (mesh_y, mesh_x))


def get_gauss_fit(prob_map, num_attempts=1, perc_mean=0.33):
    """
    Returns a gaussian fit for a given probability map
    Fitting is done via robust regression, where a fit is
    continuously refined by deleting outliers num_attempts times
    Inputs:
        prob_map [np.ndarray] set of 2-D probability map to be fit
        num_attempts: Number of times to fit & remove outliers
        perc_mean: All probability values below perc_mean*mean(gauss_fit) will be
            considered outliers for repeated attempts
    Outputs:
        gauss_fit: [np.ndarray] specifying the 2-D Gaussian PDF
        grid: [tuples] containing (y,x) points with which the Gaussian PDF can be plotted
        gauss_mean: [np.ndarray] of shape (2,) specifying the 2-D Gaussian center
        gauss_cov: [np.ndarray] of shape (2,2) specifying the 2-D Gaussian covariance matrix
    """
    assert prob_map.ndim==2, (
        'get_gauss_fit: Input prob_map must have 2 dimension specifying [num_rows, num_cols')
    if num_attempts < 1:
        num_attempts = 1
    orig_prob_map = prob_map.copy()
    gauss_success = False
    while not gauss_success:
        prob_map = orig_prob_map.copy()
        try:
            for i in range(num_attempts):
                map_min = np.min(prob_map)
                prob_map -= map_min
                map_sum = np.sum(prob_map)
                if map_sum != 1.0:
                    prob_map /= map_sum
                gauss_mean, gauss_cov = gaussian_fit(prob_map)
                gauss_fit, grid = generate_gaussian(prob_map.shape, gauss_mean, gauss_cov)
                gauss_fit = (gauss_fit * map_sum) + map_min
                if i < num_attempts-1:
                    gauss_mask = gauss_fit.copy()
                    gauss_mask[np.where(gauss_mask<perc_mean*np.mean(gauss_mask))] = 0
                    gauss_mask[np.where(gauss_mask>0)] = 1
                    prob_map *= gauss_mask
            gauss_success = True
        except np.linalg.LinAlgError: # Usually means cov matrix is singular
            print(f'get_gauss_fit: Failed to fit Gaussian at attempt {i}, trying again.'+
            '\n  To avoid this try decreasing perc_mean.')
            num_attempts = i-1
            if num_attempts <= 0:
                assert False, (
                  'get_gauss_fit: np.linalg.LinAlgError - Unable to fit gaussian.')
    return (gauss_fit, grid, gauss_mean, gauss_cov)


def construct_mask_from_mean_cov(gauss_mean, gauss_cov, shape, confidence=0.90):
    evals, evecs = np.linalg.eigh(gauss_cov)
    sort_indices = np.argsort(evals)[::-1]
    largest_eigval = evals[sort_indices][0]
    smallest_eigval = evals[sort_indices][-1]
    angle = np.arctan2(*evecs[:, sort_indices][0])
    chisq_val = scipy.stats.chi2.ppf(confidence, 2)
    height = chisq_val * np.sqrt(smallest_eigval) # b
    width = chisq_val * np.sqrt(largest_eigval) # a
    mask = np.zeros(shape)
    rr, cc = skimage.draw.ellipse(
        gauss_mean[0],
        gauss_mean[1],
        height/2, # radius
        width/2, # radius
        shape=shape,
        rotation=angle)
    mask[rr, cc] = 1
    return mask, [height, width], angle


def mask_then_normalize(vector, mask, mask_threshold):
    """
    Parameters:
        mask [np.ndarray] mask to zero out vector values with shape [vector_rows, vector_cols] or [vector_length,]
        vector [np.ndarray] vector with shape [vector_rows, vector_cols] or [vector_length,].
    Outputs:
        vector [np.ndarray] masked vector with shape [vector_length,] and l2-norm = 1
    """
    mask = mask.flatten()
    vector = vector.flatten()
    assert mask.size == vector.size, (
        f'mask size = {mask.size} must equal vector size = {vector.size}')
    mask /= mask.max()
    mask[mask<mask_threshold] = 0
    mask[mask>0] = 1
    vector = np.multiply(mask, vector)
    vector = vector / np.linalg.norm(vector)
    return vector


def poly_mask_image(image, yx_range, vector_slope, edge_length):
    ((y_min, y_max), (x_min, x_max)) = yx_range
    ranges = [y_max - y_min, x_max - x_min]
    mins = [y_min, x_min]
    cv_endpoint = y_max/vector_slope
    conv_pts = lambda li : [int((li[i] - mins[i]) / ranges[i] * edge_length) for i in range(len(li))]
    all_pts = [
        conv_pts([0, 0]), # [y, x]
        conv_pts([0, x_max]),
        conv_pts([y_max, x_max]),
        conv_pts([y_max, cv_endpoint]),
        conv_pts([0, 0])
    ]
    polygon = np.array(all_pts)
    mask = skimage.draw.polygon2mask(image.shape, polygon)
    image[~mask] = 0
    return image