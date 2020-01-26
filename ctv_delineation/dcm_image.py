"""A convenient class for handling 3D images coming from DICOM files."""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from ctv_delineation import utilities


class Plane(Enum):
    """Represents the one of three planes: axial, coronal or sagittal."""
    axial = 'axial'
    coronal = 'coronal'
    sagittal = 'sagittal'


class DICOMImage:
    """Represents a 3D image loaded from a folder of DICOM image slices.

    Attributes:
        image (np.ndarray): The underlying 3D NumPy array.
        origin(np.ndarray): A length-3 NumPy vector containing the image
            origin in real-space coordinates.
        pixel_spacing(np.ndarray): A length-3 Numpy vector containing the
            distance between pixels in mm.
    """

    def __init__(self, image, origin, pixel_spacing, slice_ids):
        self.image = image
        self.origin = origin
        self.pixel_spacing = pixel_spacing
        self.slice_ids = slice_ids
        self.aspect_ratio_axial = pixel_spacing[0] / float(pixel_spacing[1])
        self.aspect_ratio_coronal = pixel_spacing[0] / float(pixel_spacing[2])
        self.aspect_ratio_sagittal = pixel_spacing[1] / float(pixel_spacing[2])

    def get_extent(self):
        """TODO"""
        left = self.origin[0]
        top = self.origin[1]
        right = left + self.pixel_spacing[0] * self.image.shape[0]
        bottom = top + self.pixel_spacing[1] * self.image.shape[1]
        return (left, right, bottom, top)

    @classmethod
    def from_slices(cls, slices):
        """Creates a DICOMImage object from raw slices.

        Parameters:
            slices: A list of pydicom Dataset objects.

        Returns:
            A new DICOMImage.
        """
        left_right_pixel_spacing = float(slices[0].PixelSpacing[0])
        ant_post_pixel_spacing = float(slices[0].PixelSpacing[1])
        sup_inf_spacing = float(slices[0].SliceThickness)
        pixel_spacing = (left_right_pixel_spacing, ant_post_pixel_spacing,
                         sup_inf_spacing)
        # Sort slices by axial position
        slices = sorted(slices, key=lambda s: s.SliceLocation)
        slice_ids = [s.SOPInstanceUID for s in slices]
        # Create empty 3D matrix to store image
        image = np.zeros(shape=[*slices[0].pixel_array.shape, len(slices)])
        for i, s in enumerate(slices):
            image[..., i] = s.pixel_array
        origin = slices[0].ImagePositionPatient
        return cls(image, origin, pixel_spacing, slice_ids)

    def plot(self, slice_index, plane: Plane = Plane.axial, ax=plt.gca()):
        """ Plots a slice from the image using Matplotlib.

        Parameters:
            slice_index: Index of slice to plot.
            plane: A Plane enum (axial, saggital or coronal).
            ax: The Matplotlib axes object to plot to.
        """
        if plane not in Plane:
            print("{} is not a valid anatomical plane".format(plane))
            return
        if plane is Plane.axial:
            ax.imshow(self.image[:, :, slice_index], cmap='gray',
                      extent=self.get_extent())
            ax.set_aspect(self.aspect_ratio_axial)
        elif plane is Plane.coronal:
            ax.imshow(self.image[slice_index, :, :].T, cmap='gray')
            ax.set_aspect(1 / self.aspect_ratio_coronal)
        elif plane is Plane.sagittal:
            ax.imshow(np.rot90(self.image[:, slice_index, :]), cmap='gray')
            ax.set_aspect(1 / self.aspect_ratio_sagittal)

    def get_id_for_slice(self, slice_index: int):
        """Returns the slice ID for a given slice."""
        return self.slice_ids[slice_index]


def get_image_from_dicom_files(folder_name):
    """Constructs an image object from a folder containing individual DICOM slices.

    Parameters:
        folder_name (str): the name of the folder containing the DICOM files.

    Returns:
        A DICOMImage object.
    """
    dicom_files = utilities.get_files_with_extension(folder_name, ".dcm")
    slices = [pydicom.read_file(f) for f in dicom_files]
    return DICOMImage.from_slices(slices)
