import concurrent.futures
import pywt
import os
import numpy as np
import pandas as pd
from .Sera_ReadWrite import readimage, convert_modalities
import psutil
from scipy.special import factorial
import scipy.fft as fft
from scipy.ndimage import laplace
import scipy.ndimage as ndi

# Todo: m-salari: change format get spacing,origin and direction to Dict

def pool_voxel_grids(x1, x2, pooling_method):
    if pooling_method == "max":
        # Perform max pooling by selecting the maximum intensity of each voxel.
        return np.maximum(x1, x2)

    elif pooling_method == "min":
        # Perform min pooling by selecting the minimum intensity of each voxel.
        return np.minimum(x1, x2)

    elif pooling_method in ["mean", "sum"]:
        # Perform mean / sum pooling by summing the intensities of each voxel.
        return np.add(x1, x2)

    else:
        raise ValueError(f"Unknown pooling method encountered: {pooling_method}")


class SeparableFilterSet:
    def __init__(self,
                 filter_x,
                 filter_y,
                 filter_z=None,
                 pre_filter_x=None,
                 pre_filter_y=None,
                 pre_filter_z=None):
        self.x = filter_x
        self.y = filter_y
        self.z = filter_z

        self.pr_x = pre_filter_x
        self.pr_y = pre_filter_y
        self.pr_z = pre_filter_z

        # Extend even-sized filters.
        for attr in ["x", "y", "z", "pr_x", "pr_y", "pr_z"]:
            if self.__dict__[attr] is not None:

                # Check if the kernel is even or odd.
                if len(self.__dict__[attr]) % 2 == 0:
                    self.__dict__[attr] = np.append(self.__dict__[attr], 0.0)

    def permute_filters(self, rotational_invariance=True, require_pre_filter=False, as_filter_table=False):

        if require_pre_filter:
            if self.pr_x is None or self.pr_y is None:
                raise ValueError("The pre-filter should be set for all dimensions.")

            if self.z is not None and self.pr_z is None:
                raise ValueError("The pre-filter should have a component in the z-direction.")

            elif self.z is None and self.pr_z is not None:
                raise ValueError("The pre-filter should not have a component in the z-direction.")

        # Return an encapsulated version of the object.
        if not rotational_invariance:
            return [self]

        permuted_filters = []

        # Initiate filter strings
        g_x = "gx"
        g_y = "gy"
        g_z = "gz"
        jg_x = "jgx"
        jg_y = "jgy"
        jg_z = "jgz"

        # Test if x and y filters are the same.
        if np.array_equiv(self.x, self.y):
            g_y = g_x
            jg_y = jg_x

        # Test if x and z filters are the same
        if self.z is not None:
            if np.array_equiv(self.x, self.z):
                g_z = g_x
                jg_z = jg_x

        # Test if y and z filters are the same
        if self.z is not None:
            if np.array_equiv(self.y, self.z):
                g_z = g_y
                jg_z = jg_y

        # Test if the x-filter is symmetric.
        if np.array_equiv(self.x, np.flip(self.x)):
            jg_x = g_x

        # Test if the y-filter is symmetric.
        if np.array_equiv(self.y, np.flip(self.y)):
            jg_y = g_y

        # Test if the y-filter is symmetric.
        if self.z is not None:
            if np.array_equiv(self.z, np.flip(self.z)):
                jg_z = g_z

        if self.z is None:
            # 2D right-hand permutations
            permuted_filters += [{"x": g_x, "y": g_y}]
            permuted_filters += [{"x": jg_y, "y": g_x}]
            permuted_filters += [{"x": jg_x, "y": jg_y}]
            permuted_filters += [{"x": g_y, "y": jg_x}]

        else:
            # 3D right-hand permutations
            permuted_filters += [{"x": g_x, "y": g_y, "z": g_z}]
            permuted_filters += [{"x": jg_z, "y": g_y, "z": g_x}]
            permuted_filters += [{"x": jg_x, "y": g_y, "z": jg_z}]
            permuted_filters += [{"x": g_z, "y": g_y, "z": jg_x}]

            permuted_filters += [{"x": g_y, "y": g_z, "z": g_x}]
            permuted_filters += [{"x": g_y, "y": jg_z, "z": jg_x}]
            permuted_filters += [{"x": g_y, "y": jg_x, "z": g_z}]
            permuted_filters += [{"x": jg_x, "y": jg_y, "z": g_z}]

            permuted_filters += [{"x": jg_y, "y": g_x, "z": g_z}]
            permuted_filters += [{"x": jg_z, "y": jg_x, "z": g_y}]
            permuted_filters += [{"x": jg_z, "y": jg_y, "z": jg_x}]
            permuted_filters += [{"x": jg_z, "y": g_x, "z": jg_y}]

            permuted_filters += [{"x": jg_y, "y": jg_x, "z": jg_z}]
            permuted_filters += [{"x": g_x, "y": jg_y, "z": jg_z}]
            permuted_filters += [{"x": g_y, "y": g_x, "z": jg_z}]
            permuted_filters += [{"x": g_z, "y": jg_x, "z": jg_y}]

            permuted_filters += [{"x": g_z, "y": jg_y, "z": g_x}]
            permuted_filters += [{"x": g_z, "y": g_x, "z": g_y}]
            permuted_filters += [{"x": jg_x, "y": g_z, "z": g_y}]
            permuted_filters += [{"x": jg_y, "y": g_z, "z": jg_x}]

            permuted_filters += [{"x": g_x, "y": g_z, "z": jg_y}]
            permuted_filters += [{"x": jg_x, "y": jg_z, "z": jg_y}]
            permuted_filters += [{"x": jg_y, "y": jg_z, "z": g_x}]
            permuted_filters += [{"x": g_x, "y": jg_z, "z": g_y}]

        # Combine filters into a table.
        permuted_filters = pd.DataFrame(permuted_filters)

        if require_pre_filter:
            # Create a pre-filter to derive a table with filter orientations.
            pre_filter_set = SeparableFilterSet(filter_x=self.pr_x,
                                                filter_y=self.pr_y,
                                                filter_z=self.pr_z)

            permuted_pre_filters = pre_filter_set.permute_filters(rotational_invariance=rotational_invariance,
                                                                  as_filter_table=True)

            # Update the columns names
            permuted_pre_filters.rename(columns={"x": "pr_x",
                                                 "y": "pr_y",
                                                 "z": "pr_z"},
                                        inplace=True)

            # Join with the permuted_filters table.
            permuted_filters = pd.concat([permuted_pre_filters, permuted_filters], axis=1)

        if as_filter_table:
            return permuted_filters

        # Remove duplicates.
        permuted_filters = permuted_filters.drop_duplicates(ignore_index=True)

        filter_set_list = []
        for ii in range(len(permuted_filters)):
            permuted_filter_set = permuted_filters.loc[ii, :]

            filter_obj = self

            if require_pre_filter:
                if self.z is None:
                    filter_set_list += [SeparableFilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                           filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                           pre_filter_x=filter_obj._translate_filter(
                                                               permuted_filter_set.pr_x, True),
                                                           pre_filter_y=filter_obj._translate_filter(
                                                               permuted_filter_set.pr_y, True))]

                else:
                    filter_set_list += [SeparableFilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                           filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                           filter_z=filter_obj._translate_filter(permuted_filter_set.z),
                                                           pre_filter_x=filter_obj._translate_filter(
                                                               permuted_filter_set.pr_x, True),
                                                           pre_filter_y=filter_obj._translate_filter(
                                                               permuted_filter_set.pr_y, True),
                                                           pre_filter_z=filter_obj._translate_filter(
                                                               permuted_filter_set.pr_z, True))]

            else:
                if self.z is None:
                    filter_set_list += [SeparableFilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                           filter_y=filter_obj._translate_filter(
                                                               permuted_filter_set.y))]

                else:
                    filter_set_list += [SeparableFilterSet(filter_x=filter_obj._translate_filter(permuted_filter_set.x),
                                                           filter_y=filter_obj._translate_filter(permuted_filter_set.y),
                                                           filter_z=filter_obj._translate_filter(
                                                               permuted_filter_set.z))]

        return filter_set_list

    def _translate_filter(self, filter_symbol, use_pre_filter=False):

        if filter_symbol == "gx":
            if use_pre_filter:
                return self.pr_x
            else:
                return self.x

        elif filter_symbol == "gy":
            if use_pre_filter:
                return self.pr_y
            else:
                return self.y

        elif filter_symbol == "gz":
            if use_pre_filter:
                return self.pr_z
            else:
                return self.z

        elif filter_symbol == "jgx":
            if use_pre_filter:
                return np.flip(self.pr_x)
            else:
                return np.flip(self.x)

        elif filter_symbol == "jgy":
            if use_pre_filter:
                return np.flip(self.pr_y)
            else:
                return np.flip(self.y)

        elif filter_symbol == "jgz":
            if use_pre_filter:
                return np.flip(self.pr_z)
            else:
                return np.flip(self.z)

        else:
            raise ValueError(f"Encountered unrecognised filter symbol: {filter_symbol}")

    def decompose_filter(self, method="a_trous"):

        if method == "a_trous":
            # Add in 0s for the à trous algorithm

            # Iterate over filters.
            for attr in ["x", "y", "z", "pr_x", "pr_y", "pr_z"]:
                if self.__dict__[attr] is not None:
                    # Strip zeros from tail and head.
                    # old_filter_kernel = np.trim_zeros(deepcopy(self.__dict__[attr]))
                    old_filter_kernel = self.__dict__[attr]

                    # Create an array of zeros
                    new_filter_kernel = np.zeros(len(old_filter_kernel) * 2 - 1, dtype=float)

                    # Place the original filter constants at every second position. This creates a hole (0.0) between
                    # each of the filter constants.
                    new_filter_kernel[::2] = old_filter_kernel

                    # Update the attribute.
                    self.__dict__[attr] = new_filter_kernel

        else:
            raise ValueError(f"Unknown filter decomposition method: {method}")

    def convolve(self, voxel_grid, mode, use_pre_filter=False):

        # Ensure that we work from a local copy of voxel_grid to prevent updating it by reference.
        voxel_grid = voxel_grid

        if use_pre_filter:
            if self.pr_x is None or self.pr_y is None or (self.z is not None and self.pr_z is None):
                raise ValueError("Pre-filter kernels are expected, but not found.")

            # Apply filter along the z-axis. Note that the voxel grid is stored with z, y, x indexing. Hence the
            # z-axis is the first axis, the y-axis the second, and the x-axis the third.
            if self.pr_z is not None:
                voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_z, axis=0, mode=mode)

            # Apply filter along the y-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_y, axis=1, mode=mode)

            # Apply filter along the x-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.pr_x, axis=2, mode=mode)

        else:
            # Apply filter along the z-axis. Note that the voxel grid is stored with z, y, x indexing. Hence the
            # z-axis is the first axis, the y-axis the second, and the x-axis the third.
            if self.z is not None:
                voxel_grid = ndi.convolve1d(voxel_grid, weights=self.z, axis=0, mode=mode)

            # Apply filter along the y-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.y, axis=1, mode=mode)

            # Apply filter along the x-axis.
            voxel_grid = ndi.convolve1d(voxel_grid, weights=self.x, axis=2, mode=mode)

        return voxel_grid


class FilterSet:
    def __init__(self,
                 filter_set: np.ndarray,
                 transformed=False,
                 pad_image=True,
                 riesz=None,
                 riesz_steered=False):

        self.filter_set = filter_set
        self.transformed = transformed
        self.pad_image = pad_image
        self.riesz = riesz
        self.riesz_steered = riesz_steered

    def _get_coordinate_grid(self, axis=None):

        # Determine the grid center.
        grid_center = (np.array(self.filter_set.shape, dtype=np.float32) - 1.0) / 2.0

        # Determine distance from center
        coordinate_grid = list(np.indices(self.filter_set.shape, sparse=True))
        coordinate_grid = [(coordinate_grid[ii] - center_pos) / center_pos for ii, center_pos in enumerate(grid_center)]

        # Broadcast to filter size shape.
        if axis is not None:
            coordinate_grid = np.broadcast_to(coordinate_grid[axis], self.filter_set.shape)

        return coordinate_grid

    def _get_distance_grid(self):

        # Generate a coordinate grid.
        coordinate_grid = self._get_coordinate_grid()

        # Compute the distances in the grid.
        distance_grid = np.linalg.norm(coordinate_grid)

        return distance_grid

    def _pad_image(self,
                   voxel_grid,
                   mode,
                   axis=None):

        # Modes in scipy and numpy are defined differently.
        if mode == "reflect":
            mode = "symmetric"
        elif mode == "symmetric":
            mode = "reflect"
        elif mode == "nearest":
            mode = "edge"
        elif mode == "mirror":
            mode = "reflect"

        # Ensure that we work from a local copy of voxel_grid to prevent updating it by reference.
        voxel_grid = voxel_grid

        # Determine the original shape
        original_shape = voxel_grid.shape

        # Pad original image with half the kernel size on axes other than axis.
        if self.pad_image:
            pad_size = np.floor(np.array(self.filter_set.shape) / 2.0)
        else:
            pad_size = np.zeros(len(self.filter_set.shape), dtype=np.float32)

        # Determine pad widths.
        pad_width = []
        original_offset = []
        axis_id = 0

        for current_axis in range(3):
            if current_axis != axis:
                # Select current axis.
                current_pad_size = int(pad_size[axis_id])

                # Add to elements.
                pad_width.append((current_pad_size, current_pad_size))
                original_offset.append(current_pad_size)

                # Update axis_id to skip to next element.
                axis_id += 1

            else:
                pad_width.append((0, 0))
                original_offset.append(0)

        # Set padding
        voxel_grid = np.pad(voxel_grid,
                            pad_width=pad_width,
                            mode=mode)

        return voxel_grid, original_shape, original_offset

    def _transform_filter(self, filter_shape, transform_method="interpolate"):

        if transform_method not in ["zero_pad", "interpolate"]:
            raise ValueError(
                f"The transform_method argument expects \"zero_pad\" or \"interpolate\". Found: {transform_method}")

        if self.transformed and not np.equal(self.filter_set.shape, filter_shape).all() and transform_method == \
                "zero_pad":
            # Zero-padding takes place in the spatial domain. We therefore have to inverse transform the filter.
            self.filter_set = fft.ifftn(self.filter_set)

            # Transform back to the Fourier domain
            self.filter_set = fft.fftn(self.filter_set,
                                       filter_shape)

        elif self.transformed and not np.equal(self.filter_set.shape, filter_shape).all() and transform_method == \
                "interpolate":
            # Find zoom factor.
            zoom_factor = np.divide(filter_shape, self.filter_set.shape)

            # Make sure to zoom in the centric filter view.
            self.filter_set = fft.ifftshift(ndi.zoom(fft.fftshift(self.filter_set),
                                                     zoom=zoom_factor))

        elif not self.transformed:
            # Transform to the Fourier domain
            self.filter_set = fft.fftn(self.filter_set,
                                       filter_shape)

            self.transformed = True

    def _riesz_transform(self):

        # Skip if no Riesz transformation order have been set.
        if self.riesz is None:
            return

        # Skip if all Riesz transformation orders are 0.
        if not np.any(np.array(self.riesz) > 0):
            return

        # Check if the number of dimensions match.
        if len(self.riesz) is not np.ndim(self.filter_set):
            raise ValueError(f"The number of transformation orders ({len(self.riesz)}) does not match the filter "
                             f"dimension ({np.ndim(self.filter_set)}).")

        # Determine the order sum (L).
        order_sum = np.sum(np.array(self.riesz))

        # Compute the pre-factor, see equation 4.11.
        prefactor = (-1.0j) ** order_sum * np.sqrt(factorial(order_sum) / np.prod(factorial(np.array(self.riesz))))

        # Distance grid, see equation 4.11.
        distance_grid = np.power(self._get_distance_grid(), order_sum)

        # Set up gradient grid.
        gradient_grid = np.ones(self.filter_set.shape, dtype=np.float32)

        # Iterate over transformation orders.
        for ii, riesz_transform_order in enumerate(self.riesz):
            # Skip if no transformation is required.
            if riesz_transform_order == 0:
                continue

            # Update the gradient grid.
            gradient_grid = np.multiply(gradient_grid,
                                        np.power(self._get_coordinate_grid(axis=ii), riesz_transform_order))

        # Divide by l2-norm.
        gradient_grid = prefactor * np.divide(gradient_grid, distance_grid)

        # Update filter by taking Hadamard product.
        self.filter_set = fft.ifftshift(np.multiply(fft.fftshift(self.filter_set), gradient_grid))

    def _return_response(self, voxel_grid, response, original_offset, original_shape):
        # Compute response map.
        if response in ["modulus", "abs", "magnitude"]:
            voxel_grid = np.abs(voxel_grid)

        elif response in ["angle", "phase", "argument"]:
            voxel_grid = np.angle(voxel_grid)

        elif response in ["real"]:
            voxel_grid = np.real(voxel_grid)

        elif response in ["imaginary"]:
            voxel_grid = np.imag(voxel_grid)

        else:
            raise ValueError(f"The response argument should be \"modulus\", \"abs\", \"magnitude\", \"angle\", "
                             f"\"phase\", \"argument\", \"real\" or \"imaginary\". Found: {response}")

        # Crop image to original size.
        voxel_grid = voxel_grid[2 * original_offset[0]:2 * original_offset[0] + original_shape[0],
                     2 * original_offset[1]:2 * original_offset[1] + original_shape[1],
                     2 * original_offset[2]:2 * original_offset[2] + original_shape[2]]

        return voxel_grid


class FilterSet2D(FilterSet):

    def convolve(self,
                 voxel_grid: np.ndarray,
                 mode,
                 response,
                 axis=0):
        # Pad the image prior to convolution so that the valid convolution spans the image.
        voxel_grid, original_shape, original_offset = self._pad_image(voxel_grid, mode=mode, axis=axis)

        # Determine the filter output shape.
        filter_output_shape = [dim_size for current_axis, dim_size in enumerate(voxel_grid.shape) if
                               current_axis != axis]

        # Compute the fft of the filter, with output shape of the image.
        self._transform_filter(filter_shape=filter_output_shape)

        # Riesz transform the filter.
        self._riesz_transform()

        # Iterate over slices, compute fft, multiply with filter, and compute inverse fourier transform.
        voxel_grid = np.stack([self._convolve(voxel_grid=np.squeeze(current_grid, axis=axis))
                               for current_grid in np.split(voxel_grid, voxel_grid.shape[axis], axis=axis)],
                              axis=axis)

        return self._return_response(voxel_grid=voxel_grid,
                                     response=response,
                                     original_offset=original_offset,
                                     original_shape=original_shape)

    def _convolve(self,
                  voxel_grid: np.ndarray):
        # Compute FFT of the slice.
        f_voxel = fft.fft2(voxel_grid)

        if not self.transformed:
            raise ValueError("Filter should have been transformed to the Fourier domain.")

        # Multiply with the filter and return the inverse fourier transform of the hadamard product.
        return fft.ifft2(f_voxel * self.filter_set)


class FilterSet3D(FilterSet):

    def convolve(self,
                 voxel_grid: np.ndarray,
                 mode,
                 response):
        # Pad the image prior to convolution so that the valid convolution spans the image.
        voxel_grid, original_shape, original_offset = self._pad_image(voxel_grid, mode=mode)

        # Determine the filter output shape.
        filter_output_shape = [dim_size for current_axis, dim_size in enumerate(voxel_grid.shape)]

        # Compute the fft of the filter, with output shape of the image.
        self._transform_filter(filter_shape=filter_output_shape)

        # Riesz transform the filter.
        self._riesz_transform()

        # Iterate over slices, compute fft, multiply with filter, and compute inverse fourier transform.
        voxel_grid = self._convolve(voxel_grid)

        return self._return_response(voxel_grid=voxel_grid,
                                     response=response,
                                     original_offset=original_offset,
                                     original_shape=original_shape)

    def _convolve(self,
                  voxel_grid: np.ndarray):
        # Compute FFT of the slice
        f_voxel = fft.fftn(voxel_grid)

        if not self.transformed:
            raise ValueError("Filter should have been transformed to the Fourier domain.")

        # Multiply with the filter and return the inverse fourier transform of the hadamard product.
        return fft.ifftn(f_voxel * self.filter_set)


def mean_filter(data, by_slice, BoundaryCondition, FilterSize):
    filter_size = FilterSize
    filter_kernel = np.ones(filter_size, dtype=np.float32) / filter_size
    if by_slice:
        filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel)
    else:
        filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel, filter_z=filter_kernel)

    voxel_grid = filter_set.convolve(voxel_grid=data, mode=BoundaryCondition)
    return voxel_grid


def mean_filter_folder(sourcefolder, by_slice, BoundaryCondition, FilterSize, destfolder):
    Filtered_data = os.listdir(sourcefolder)
    fixed = [i for i in Filtered_data if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            Fixed_fullpath = os.path.join(sourcefolder, co)
            fixed_img = readimage(Fixed_fullpath)
            try:
                if isinstance(fixed_img[0], np.ndarray):
                    if len(fixed_img[0].shape) == 3:
                        futures.append(executor.submit(mean_filter_folder_Thread,
                                                       fixed_img, by_slice, BoundaryCondition, FilterSize, destfolder
                                                       ))

                    else:
                        raise('Images must be 3D.')
                else:
                    raise('You must use an approprate type of input.')
            except Exception as e:
                raise('Out of Memory or the parameters of filtering tool should be selected properly:', e)
    executor.shutdown(wait=True)

    return ""


def mean_filter_folder_Thread(img, by_slice, BoundaryCondition, FilterSize, destfolder):
    # filter_size = FilterSize
    # filter_kernel = np.ones(filter_size, dtype=np.float32) / filter_size

    # if by_slice:
    #     filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel)
    # else:
    #     filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel, filter_z=filter_kernel)

    # voxel_grid = filter_set.convolve(voxel_grid=img[0],mode=BoundaryCondition)

    voxel_grid = mean_filter(img[0], by_slice, BoundaryCondition, FilterSize)

    if (img[2] == "Dicom"):
        img[2] = "SDicom"

    convert_modalities(voxel_grid, img[1], img[2], img[2], destfolder, img[3], createfolder='False')

    return ''


def RunLoGFilter(data, spacing,
                 sigma,
                 img_average,
                 sigma_cutoff,
                 mode,
                 riesz,
                 riesz_steered,
                 by_slice
                 ):
    if sigma == 0.0:
        voxel_grid = laplace(data, mode=mode)

        return voxel_grid

    elif sigma > 0.0:

        vox_sigma = np.divide(np.full(shape=(3), fill_value=sigma), spacing)

        voxel_grid = Log_transform_grid(voxel_grid=data, sigma=vox_sigma,
                                        sigma_cutoff=sigma_cutoff,
                                        by_slice=by_slice,
                                        riesz=riesz,
                                        riesz_steered=riesz_steered,
                                        mode=mode)

        return voxel_grid


def Log_transform_grid(voxel_grid, sigma, sigma_cutoff, by_slice,
                       riesz,
                       riesz_steered,
                       mode
                       ):
    filter_size = 1 + 2 * np.floor(sigma_cutoff * sigma + 0.5)
    filter_size.astype(np.int32)

    if by_slice:
        D = 2.0

        y, x = np.mgrid[:filter_size[1], :filter_size[2]]
        y -= (filter_size[1] - 1.0) / 2.0
        x -= (filter_size[2] - 1.0) / 2.0

        norm_2 = np.power(y, 2.0) + np.power(x, 2.0)

    else:
        D = 3.0

        z, y, x = np.mgrid[:filter_size[0], :filter_size[1], :filter_size[2]]
        z -= (filter_size[0] - 1.0) / 2.0
        y -= (filter_size[1] - 1.0) / 2.0
        x -= (filter_size[2] - 1.0) / 2.0

        norm_2 = np.power(z, 2.0) + np.power(y, 2.0) + np.power(x, 2.0)

    sigma = np.max(sigma)

    scale_factor = - 1.0 / sigma ** 2.0 * np.power(1.0 / np.sqrt(2.0 * np.pi * sigma ** 2), D) * (D - norm_2 /
                                                                                                  sigma ** 2.0)
    width_factor = - norm_2 / (2.0 * sigma ** 2.0)

    filter_weights = np.multiply(scale_factor, np.exp(width_factor))

    if by_slice:

        log_filter = FilterSet2D(filter_weights,
                                 riesz=riesz,
                                 riesz_steered=riesz_steered)

        response_map = log_filter.convolve(voxel_grid=voxel_grid,
                                           mode=mode,
                                           response="real")

    else:

        log_filter = FilterSet3D(filter_weights,
                                 riesz=riesz,
                                 riesz_steered=riesz_steered)

        # Convolve laplacian of gaussian filter with the image.
        response_map = log_filter.convolve(voxel_grid=voxel_grid,
                                           mode=mode,
                                           response="real")

    return response_map

    # "modulus", "abs", "magnitude"
    # "angle", "phase", "argument"
    # "real"
    # "imaginary"


def log_filter(img, by_slice, BoundaryCondition, img_average, riesz_steered, sigma, sigma_cutoff, riesz):
    data = img[0]
    header = img[1]

        # spacing = (abs(header.affine[0,0]),abs(header.affine[1,1]),abs(header.affine[2,2]))
    # spacing = np.array(header.GetSpacing())[::-1]


    # if img[2] == 'Nifti':
    #     spacing = np.array(header.GetSpacing())[::-1]
    # elif img[2] == 'Nrrd':
    #     spacing = np.array((header['space directions'][0,0],header['space directions'][1,1],header['space directions'][2,2]))
    # elif img[2] == 'Dicom':
    #     spacing = np.array(header.GetSpacing())

    spacing = np.array(header['spacing'])
    mode = BoundaryCondition

    voxel_grid = RunLoGFilter(data, spacing,
                              sigma,
                              img_average,
                              sigma_cutoff,
                              mode,
                              riesz,
                              riesz_steered,
                              by_slice
                              )

    return voxel_grid


def log_filter_folder(sourcefolder, by_slice, BoundaryCondition, CalculateAverage, Riesz_Steered, Sigma, SigmaTruncate,
                      Riesz, destfolder):
    Filtered_data = os.listdir(sourcefolder)
    fixed = [i for i in Filtered_data if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            Fixed_fullpath = os.path.join(sourcefolder, co)
            fixed_img = readimage(Fixed_fullpath)
            try:
                if isinstance(fixed_img[0], np.ndarray):
                    if len(fixed_img[0].shape) == 3:
                        futures.append(executor.submit(log_filter_folder_Thread,
                                                       fixed_img, by_slice, BoundaryCondition,
                                                       CalculateAverage, Riesz_Steered, Sigma, SigmaTruncate, Riesz,
                                                       destfolder
                                                       ))

                    else:
                        raise('Images must be 3D.')
                else:
                    raise('You must use an approprate type of input.')
            except Exception as e:
                raise('Out of Memory or the parameters of filtering tool should be selected properly:', e)
    executor.shutdown(wait=True)

    return ""


def log_filter_folder_Thread(img, by_slice, BoundaryCondition, img_average, riesz_steered, sigma, sigma_cutoff, riesz,
                             destfolder):
    voxel_grid = log_filter(img, by_slice, BoundaryCondition, img_average, riesz_steered, sigma, sigma_cutoff, riesz)

    if (img[2] == "Dicom"):
        img[2] = "SDicom"

    convert_modalities(voxel_grid, img[1], img[2], img[2], destfolder, img[3], createfolder='False')

    return ''


def get_filter_set(kernels, kernel_normalise):
    # Deparse kernels to a list
    kernel_list = [kernels[ii:ii + 2] for ii in range(0, len(kernels), 2)]

    filter_x = None
    filter_y = None
    filter_z = None

    for ii, kernel in enumerate(kernel_list):
        if kernel.lower() == "l5":
            laws_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif kernel.lower() == "e5":
            laws_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
        elif kernel.lower() == "s5":
            laws_kernel = np.array([-1.0, 0.0, 2.0, 0.0, -1.0])
        elif kernel.lower() == "w5":
            laws_kernel = np.array([-1.0, 2.0, 0.0, -2.0, 1.0])
        elif kernel.lower() == "r5":
            laws_kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
        elif kernel.lower() == "l3":
            laws_kernel = np.array([1.0, 2.0, 1.0])
        elif kernel.lower() == "e3":
            laws_kernel = np.array([-1.0, 0.0, 1.0])
        elif kernel.lower() == "s3":
            laws_kernel = np.array([-1.0, 2.0, -1.0])
        else:
            raise ValueError(f"{kernel} is not an implemented Laws kernel")

        # Normalise kernel
        if kernel_normalise:
            laws_kernel /= np.sqrt(np.sum(np.power(laws_kernel, 2.0)))

        # Assign filter to variable.
        if ii == 0:
            filter_x = laws_kernel
        elif ii == 1:
            filter_y = laws_kernel
        elif ii == 2:
            filter_z = laws_kernel

    # Create FilterSet object
    return SeparableFilterSet(filter_x=filter_x,
                              filter_y=filter_y,
                              filter_z=filter_z)


def response_to_energy(voxel_grid, delta, energy_normalise, by_slice, mode):
    # Take absolute value of the voxel grid.
    voxel_grid = np.abs(voxel_grid)

    # Set the filter size.
    filter_size = 2 * delta + 1

    # Set up the filter kernel.
    if energy_normalise:
        filter_kernel = np.ones(filter_size, dtype=np.float32) / filter_size
    else:
        filter_kernel = np.ones(filter_size, dtype=np.float32)

    # Create a filter set.
    if by_slice:
        filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel)
    else:
        filter_set = SeparableFilterSet(filter_x=filter_kernel, filter_y=filter_kernel, filter_z=filter_kernel)

    # Apply the filter.
    voxel_grid = filter_set.convolve(voxel_grid=voxel_grid, mode=mode)

    return voxel_grid


def Laws_filter(img, by_slice, BoundaryCondition, Kernel, pooling_method, cal_energy, rotation_inver, delta):
    kernel_normalise = True
    energy_normalise = True

    name = Kernel
    mode = BoundaryCondition
    data = img[0]
    # header = img[1]
    rot_invariance = rotation_inver
    calculate_energy = cal_energy

    filter_set = get_filter_set(kernels=name, kernel_normalise=kernel_normalise)
    filter_list = filter_set.permute_filters(rotational_invariance=rot_invariance)

    img_voxel_grid = np.zeros(data.shape, dtype=np.float32)

    for ii, filter_set in enumerate(filter_list):
        # Convolve and compute response map.
        img_laws_grid = filter_set.convolve(voxel_grid=data,
                                            mode=mode)
        # Perform pooling
        if ii == 0:
            # Initially, update img_voxel_grid.
            img_voxel_grid = img_laws_grid
        else:
            # Pool grids.
            img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_laws_grid,
                                              pooling_method=pooling_method)
        # Remove img_laws_grid to explicitly release memory when collecting garbage.
        del img_laws_grid

    if pooling_method == "mean":
        # Perform final pooling step for mean pooling.
        img_voxel_grid = np.divide(img_voxel_grid, len(filter_list))

    # Compute energy map from the response map.
    if calculate_energy:
        img_voxel_grid = response_to_energy(voxel_grid=img_voxel_grid,
                                            delta=delta,
                                            energy_normalise=energy_normalise,
                                            by_slice=by_slice,
                                            mode=mode
                                            )

    return img_voxel_grid


def Laws_filter_folder(sourcefolder, by_slice, BoundaryCondition, Kernel, pooling_method, cal_energy, rotation_inver,
                       delta, destfolder):
    Filtered_data = os.listdir(sourcefolder)
    fixed = [i for i in Filtered_data if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            Fixed_fullpath = os.path.join(sourcefolder, co)
            fixed_img = readimage(Fixed_fullpath)
            try:
                if isinstance(fixed_img[0], np.ndarray):
                    if len(fixed_img[0].shape) == 3:
                        futures.append(executor.submit(Laws_filter_folder_Thread,
                                                       fixed_img, by_slice, BoundaryCondition, Kernel, pooling_method,
                                                       cal_energy, rotation_inver, delta, destfolder
                                                       ))

                    else:
                        raise('Images must be 3D.')
                else:
                    raise('You must use an approprate type of input.')
            except Exception as e:
                raise('Out of Memory or the parameters of filtering tool should be selected properly:', e)
    executor.shutdown(wait=True)

    return ""


def Laws_filter_folder_Thread(img, by_slice, BoundaryCondition, Kernel, pooling_method, cal_energy, rotation_inver,
                              delta, destfolder):
    voxel_grid = Laws_filter(img, by_slice, BoundaryCondition, Kernel, pooling_method, cal_energy, rotation_inver,
                             delta)

    if (img[2] == "Dicom"):
        img[2] = "SDicom"

    convert_modalities(voxel_grid, img[1], img[2], img[2], destfolder, img[3], createfolder='False')

    return ''


def Gabortransform_grid(voxel_grid: np.ndarray,
                        sigma: np.float32,
                        gamma: np.float32,
                        lamda: np.float32,
                        theta: np.float32,
                        # filter_size,
                        stack_axis,
                        sigma_cutoff,
                        riesz,
                        riesz_steered,
                        mode,
                        response_type
                        ):
    if sigma_cutoff is not None:
        alpha = sigma_cutoff * sigma
        beta = sigma_cutoff * sigma * gamma

        x_size = max(np.abs(alpha * np.cos(theta) + beta * np.sin(theta)),
                     np.abs(-alpha * np.cos(theta) + beta * np.sin(theta)),
                     1)
        y_size = max(np.abs(alpha * np.sin(theta) - beta * np.cos(theta)),
                     np.abs(-alpha * np.sin(theta) - beta * np.cos(theta)),
                     1)

        x_size = int(1 + 2 * np.floor(x_size + 0.5))
        y_size = int(1 + 2 * np.floor(y_size + 0.5))

    else:
        x_size = voxel_grid.shape[2]
        y_size = voxel_grid.shape[1]

        x_size = int(1 + 2 * np.floor(x_size / 2.0))
        y_size = int(1 + 2 * np.floor(y_size / 2.0))

    y, x = np.mgrid[:y_size, :x_size].astype(np.float32)
    y -= (y_size - 1.0) / 2.0
    x -= (x_size - 1.0) / 2.0

    rotation_matrix = np.array([[-np.cos(theta), np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    rotated_scan_coordinates = np.dot(rotation_matrix, np.array((y.flatten(), x.flatten())))
    y = rotated_scan_coordinates[0, :].reshape((y_size, x_size))
    x = rotated_scan_coordinates[1, :].reshape((y_size, x_size))

    gabor_filter = np.exp(-(np.power(x, 2.0) + gamma ** 2.0 * np.power(y, 2.0)) / (2.0 * sigma ** 2.0) + 1.0j * (
            2.0 * np.pi * x) / lamda)

    gabor_filter = FilterSet2D(gabor_filter,
                               riesz=riesz,
                               riesz_steered=riesz_steered)

    response_map = gabor_filter.convolve(voxel_grid=voxel_grid,
                                         mode=mode,
                                         response=response_type,
                                         axis=stack_axis)

    return response_map


def gabor_filter(img, by_slice, BoundaryCondition, pooling_method, response, rotation_inver, gamma, lambdaa, sigma,
                 step, theta_initial, SigmaTruncate):
    data = img[0]
    header = img[1]

    #             spacing = (abs(header.affine[0,0]),abs(header.affine[1,1]),abs(header.affine[2,2]))
    # spacing = np.array(header.GetSpacing())[::-1]
    # if img[2] == 'Nifti':
    #     spacing = np.array(header.GetSpacing())[::-1]
    # elif img[2] == 'Nrrd':
    #     spacing = np.array((header['space directions'][0,0],header['space directions'][1,1],header['space directions'][2,2]))
    # elif img[2] == 'Dicom':
    #     spacing = np.array(header.GetSpacing())

    spacing = np.array(header['spacing'])

    mode = BoundaryCondition
    rot_invariance = rotation_inver
    sigma_cutoff = SigmaTruncate
    lambda_parameter = lambdaa
    theta = theta_initial
    theta_step = step
    response_type = response

    riesz = None
    riesz_steered = False

    if by_slice or not rot_invariance:
        sigma = np.max(np.divide(np.full(shape=(2), fill_value=sigma), spacing[:2]))
        lamda = np.max(np.divide(np.full(shape=(2), fill_value=lambda_parameter), spacing[:2]))
    else:
        sigma = np.max(np.divide(np.full(shape=(3), fill_value=sigma), spacing))
        lamda = np.max(np.divide(np.full(shape=(3), fill_value=lambda_parameter), spacing))

    if theta_step > 0.0:
        theta = theta + np.arange(start=0.0, stop=2.0, step=theta_step)
        theta = theta.tolist()
    else:
        theta = [theta]

    if by_slice or not rot_invariance:
        stack_axis = [0]
    else:
        stack_axis = [0, 1, 2]

    # Create empty voxel grid
    img_voxel_grid = np.zeros(data.shape, dtype=np.float32)

    for jj, current_axis in enumerate(stack_axis):
        for ii, current_theta in enumerate(theta):

            # Create filter and compute response map.
            img_gabor_grid = Gabortransform_grid(voxel_grid=data,
                                                 sigma=sigma,
                                                 gamma=gamma,
                                                 lamda=lamda,
                                                 theta=current_theta * np.pi,
                                                 # filter_size=filter_size,
                                                 stack_axis=current_axis,
                                                 sigma_cutoff=sigma_cutoff,
                                                 riesz=riesz,
                                                 riesz_steered=riesz_steered,
                                                 mode=mode,
                                                 response_type=response_type
                                                 )

            if ii == jj == 0:
                img_voxel_grid = img_gabor_grid
            else:
                img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_gabor_grid,
                                                  pooling_method=pooling_method)
            del img_gabor_grid

    if pooling_method == "mean":
        img_voxel_grid = np.divide(img_voxel_grid, len(stack_axis) * len(theta))

    return img_voxel_grid


def gabor_filter_folder(sourcefolder, by_slice, BoundaryCondition, pooling_method, response, rotation_inver, gamma,
                        lambdaa, sigma, step, theta_initial, SigmaTruncate, destfolder):
    Filtered_data = os.listdir(sourcefolder)
    fixed = [i for i in Filtered_data if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            Fixed_fullpath = os.path.join(sourcefolder, co)
            fixed_img = readimage(Fixed_fullpath)
            try:
                if isinstance(fixed_img[0], np.ndarray):
                    if len(fixed_img[0].shape) == 3:
                        futures.append(executor.submit(gabor_filter_folder_Thread,
                                                       fixed_img,
                                                       by_slice, BoundaryCondition, pooling_method, response,
                                                       rotation_inver, gamma, lambdaa, sigma, step, theta_initial,
                                                       SigmaTruncate
                                                       , destfolder
                                                       ))

                    else:
                        raise('Images must be 3D.')
                else:
                    raise('You must use an approprate type of input.')
            except Exception as e:
                raise('Out of Memory or the parameters of filtering tool should be selected properly:', e)
    executor.shutdown(wait=True)

    return ""


def gabor_filter_folder_Thread(img, by_slice, BoundaryCondition, pooling_method, response, rotation_inver, gamma,
                               lambdaa, sigma, step, theta_initial, SigmaTruncate, destfolder):
    voxel_grid = gabor_filter(img, by_slice, BoundaryCondition, pooling_method, response, rotation_inver, gamma,
                              lambdaa, sigma, step, theta_initial, SigmaTruncate)

    if (img[2] == "Dicom"):
        img[2] = "SDicom"

    convert_modalities(voxel_grid, img[1], img[2], img[2], destfolder, img[3], createfolder='False')

    return ''


def get_filter_set_wavelet(filter_configuration, wavelet_family):
    kernel_list = [filter_configuration[ii:ii + 1] for ii in range(0, len(filter_configuration), 1)]

    filter_x, filter_y, filter_z = None, None, None
    pre_filter_x, pre_filter_y, pre_filter_z = None, None, None

    pre_filter_kernel = np.array(pywt.Wavelet(wavelet_family).dec_lo)

    for ii, kernel in enumerate(kernel_list):
        if kernel.lower() == "l":
            wavelet_kernel = np.array(pywt.Wavelet(wavelet_family).dec_lo)
        elif kernel.lower() == "h":
            wavelet_kernel = np.array(pywt.Wavelet(wavelet_family).dec_hi)
        else:
            raise ValueError(f"{kernel} was not recognised as the component of a separable wavelet filter. It "
                             f"should be L or H.")

        if ii == 0:
            filter_x = wavelet_kernel
            pre_filter_x = pre_filter_kernel
        elif ii == 1:
            filter_y = wavelet_kernel
            pre_filter_y = pre_filter_kernel
        elif ii == 2:
            filter_z = wavelet_kernel
            pre_filter_z = pre_filter_kernel

    # Create FilterSet object
    return SeparableFilterSet(filter_x=filter_x,
                              filter_y=filter_y,
                              filter_z=filter_z,
                              pre_filter_x=pre_filter_x,
                              pre_filter_y=pre_filter_y,
                              pre_filter_z=pre_filter_z)


class NonSeparableWavelet:
    def __init__(self,
                 by_slice,
                 mode,
                 wavelet_family,
                 filter_size,
                 riesz=None,
                 riesz_steered=False,
                 response="real"):
        self.by_slice = by_slice
        self.wavelet_family = wavelet_family
        self.filter_size = filter_size
        self.max_frequency = 1.0
        self.mode = mode
        self.riesz = riesz
        self.riesz_steered = riesz_steered
        self.response = response

    def shannon_filter(self, decomposition_level=1, filter_size=None):
        """
        Set up the shannon filter in the Fourier domain.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(decomposition_level=decomposition_level,
                                                              filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=np.float32)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 2.0, distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += 1.0

        return wavelet_filter

    def simoncelli_filter(self, decomposition_level=1, filter_size=None):
        """
        Set up the simoncelli filter in the Fourier domain.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """

        # Get the distance grid.
        distance_grid, max_frequency = self.get_distance_grid(decomposition_level=decomposition_level,
                                                              filter_size=filter_size)

        # Set up a wavelet filter for the decomposition specifically.
        wavelet_filter = np.zeros(distance_grid.shape, dtype=np.float32)

        # Set the mask for the filter.
        mask = np.logical_and(distance_grid >= max_frequency / 4.0,
                              distance_grid <= max_frequency)

        # Update the filter.
        wavelet_filter[mask] += np.cos(np.pi / 2.0 * np.log2(2.0 * distance_grid[mask] / max_frequency))

        return wavelet_filter

    def get_distance_grid(self, decomposition_level=1, filter_size=None):
        """
        Create the distance grid.
        @param decomposition_level: Decomposition level for the filter.
        @param filter_size: Size of the filter. By default equal to the size of the image.
        """
        # Set up filter shape
        if filter_size is not None:
            self.filter_size = np.array(filter_size)
            if self.by_slice:
                filter_shape = (self.filter_size[1], self.filter_size[2])
            else:
                filter_shape = (self.filter_size[0], self.filter_size[1], self.filter_size[2])
        else:
            if self.by_slice:
                filter_shape = (self.filter_size, self.filter_size)

            else:
                filter_shape = (self.filter_size, self.filter_size, self.filter_size)

        # Determine the grid center.
        grid_center = (np.array(filter_shape, dtype=np.float32) - 1.0) / 2.0

        # Determine distance from center
        distance_grid = list(np.indices(filter_shape, sparse=True))
        distance_grid = [(distance_grid[ii] - center_pos) / center_pos for ii, center_pos in enumerate(grid_center)]

        # Compute the distances in the grid.
        distance_grid = np.linalg.norm(distance_grid)

        # Set the Nyquist frequency
        decomposed_max_frequency = self.max_frequency / 2.0 ** (decomposition_level - 1.0)

        return distance_grid, decomposed_max_frequency

    def convolve(self, voxel_grid, decomposition_level=1):

        # Create the kernel.
        if self.wavelet_family == "simoncelli":
            wavelet_kernel_f = self.simoncelli_filter(decomposition_level=decomposition_level,
                                                      filter_size=voxel_grid.shape)
        elif self.wavelet_family == "shannon":
            wavelet_kernel_f = self.shannon_filter(decomposition_level=decomposition_level,
                                                   filter_size=voxel_grid.shape)
        else:
            raise ValueError(f"The specified wavelet family is not implemented: {self.wavelet_family}")

        if self.by_slice:
            # Create filter set, and assign wavelet filter. Note the ifftshift that is present to go from a centric
            # to quadrant FFT representation.
            filter_set = FilterSet2D(filter_set=fft.ifftshift(wavelet_kernel_f),
                                     transformed=True,
                                     pad_image=False,
                                     riesz=self.riesz,
                                     riesz_steered=self.riesz_steered)

            # Create the response map.
            response_map = filter_set.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response=self.response,
                                               axis=0)
        else:
            # Create filter set, and assign wavelet filter. Note the ifftshift that is present to go from a centric
            # to quadrant FFT representation.
            filter_set = FilterSet3D(filter_set=fft.ifftshift(wavelet_kernel_f),
                                     transformed=True,
                                     pad_image=False,
                                     riesz=self.riesz,
                                     riesz_steered=self.riesz_steered)

            # Create the response map.
            response_map = filter_set.convolve(voxel_grid=voxel_grid,
                                               mode=self.mode,
                                               response=self.response)

        return response_map


def Wavelet_filter(img, by_slice, mode, filter_config, pooling_method, wavelet_family, riesz_steered, rot_invariance,
                   decomposition_level, filter_size, riesz, wavelet_type):
    data = img[0]
    # header = img[1]

    # stationary_wavelet = False   ####
    
    is_separable = wavelet_type in pywt.wavelist(kind="discrete")

    if is_separable:

        img_voxel_grid = np.zeros(data.shape, dtype=np.float32)

        # main_filter_set = get_filter_set_wavelet(filter_configuration=filter_config, wavelet_family=wavelet_family)
        main_filter_set = get_filter_set_wavelet(filter_configuration=filter_config, wavelet_family=wavelet_type)

        filter_list = main_filter_set.permute_filters(rotational_invariance=rot_invariance,
                                                      require_pre_filter=decomposition_level > 1)

        # Iterate over the filters.
        for ii, filter_set in enumerate(filter_list):

            for decomp_level in np.arange(decomposition_level):

                use_pre_filter = decomp_level < decomposition_level - 1

                img_wavelet_grid = filter_set.convolve(voxel_grid=data,
                                                       mode=mode,
                                                       use_pre_filter=use_pre_filter)

                if use_pre_filter:
                    filter_set.decompose_filter()

            if ii == 0:
                img_voxel_grid = img_wavelet_grid
            else:
                img_voxel_grid = pool_voxel_grids(x1=img_voxel_grid, x2=img_wavelet_grid,
                                                  pooling_method=pooling_method)
                del img_wavelet_grid

        if pooling_method == "mean":
            img_voxel_grid = np.divide(img_voxel_grid, len(filter_list))

        return img_voxel_grid

    else:

        if wavelet_family in ["simoncelli", "shannon"]:
            filter_set = NonSeparableWavelet(by_slice=by_slice,
                                             mode=mode,
                                             wavelet_family=wavelet_family,
                                             filter_size=filter_size,
                                             riesz=riesz,
                                             riesz_steered=riesz_steered)

        else:
            raise ValueError(f"{wavelet_family} is not a known separable wavelet.")

        img_wavelet_grid = filter_set.convolve(voxel_grid=data,
                                               decomposition_level=decomposition_level)

        return img_wavelet_grid


def Wavelet_filter_folder(sourcefolder, by_slice, mode, filter_config, pooling_method, wavelet_family, riesz_steered,
                          rot_invariance, decomposition_level, filter_size, riesz, wavelet_type, destfolder):
    Filtered_data = os.listdir(sourcefolder)
    fixed = [i for i in Filtered_data if (
                i.endswith(".nrrd") | i.endswith(".dcm") | i.endswith(".dicom") | i.endswith(".nii") | i.endswith(
            ".nii.gz"))]

    Num_worker = int(psutil.virtual_memory()[1] / pow(10, 9) / 2)
    if Num_worker == 0:
        Num_worker = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=Num_worker) as executor:
        futures = []
        for co in fixed:
            Fixed_fullpath = os.path.join(sourcefolder, co)
            fixed_img = readimage(Fixed_fullpath)
            try:
                if isinstance(fixed_img[0], np.ndarray):
                    if len(fixed_img[0].shape) == 3:
                        futures.append(executor.submit(Wavelet_filter_folder_Thread,
                                                       fixed_img,
                                                       by_slice, mode, filter_config, pooling_method, wavelet_family,
                                                       riesz_steered, rot_invariance, decomposition_level, filter_size,
                                                       riesz, wavelet_type
                                                       , destfolder
                                                       ))

                    else:
                        raise('Images must be 3D.')
                else:
                    raise('You must use an approprate type of input.')
            except Exception as e:
                raise('Out of Memory or the parameters of filtering tool should be selected properly:', e)
    executor.shutdown(wait=True)

    return ""


def Wavelet_filter_folder_Thread(img, by_slice, mode, filter_config, pooling_method, wavelet_family, riesz_steered,
                                 rot_invariance, decomposition_level, filter_size, riesz, wavelet_type, destfolder):

    voxel_grid = Wavelet_filter(img, by_slice, mode, filter_config, pooling_method, wavelet_family, riesz_steered,
                                rot_invariance, decomposition_level, filter_size, riesz, wavelet_type)

    if (img[2] == "Dicom"):
        img[2] = "SDicom"

    convert_modalities(voxel_grid, img[1], img[2], img[2], destfolder, img[3], createfolder='False')

    return ''
