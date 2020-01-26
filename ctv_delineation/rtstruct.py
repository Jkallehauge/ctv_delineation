from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pydicom


class RTStructFile:
    """A wrapper to make it easier to work with DICOM RTSTRUCT files.

    Attributes:
        structures (List[Structure]): A list of Structures contained in the
            RTSTRUCT file.
    """

    def __init__(self, file_name: str):
        """Creates the object.

        Args:
            file_name (str): File name of the RTSTRUCT file to load.
        """
        rtstruct = pydicom.read_file(file_name)
        self.structures = []
        roi_names = [rtstruct.StructureSetROISequence[i].ROIName for
                     i in range(len(rtstruct.ROIContourSequence))]
        for name, roi in zip(roi_names, rtstruct.ROIContourSequence):
            self.structures.append(Structure(name, roi))


class Structure:
    """Represents a single structure from an RTSTRUCT DICOM file.

    Attributes:
        name (str): The name of the structure.
        color (tuple): An (R, G, B) color tuple assigned to the structure.
        number (int): A unique identifier for the structure.
        contours (List[Contour]): List of Contour objects which comprise this
            structure.
    """

    def __init__(self, name: str, roi: pydicom.Dataset):
        """Initialize the structure.

        Args:
            name (str): The name of the structure.
            roi (pydicom.Dataset): A single region of interest from the
            ROIContourSequence DICOM tag.
        """
        self.name = name
        self.color = roi.ROIDisplayColor
        self.number = roi.ReferencedROINumber
        self.contours = []
        for contour in roi.ContourSequence:
            self.contours.append(Contour(contour, name, self.color))

    def __str__(self):
        string = ("Structure {}\n"
                  "Color: {}\n"
                  "Number of contours: {}")
        return string.format(self.name, self.color, len(self.contours))


class Contour:
    """A single contour, many of which comprise one Structure object.

    Attributes:
        image_id (int): The unique ID of the image this contour was defined
            on.
        name (str): The name of the structure this contour is part of.
        color (np.ndarray): The color used to create this contour in the
            planning system.
        num_points (int): The number of distinct points comprising this
            contour.
        points (np.ndarray): A (num_points x 3)-dimensional NumPy array of
            points comprising this contour.
    """

    def __init__(self, contour: pydicom.Dataset, name: str, color: np.ndarray):
        self.num_points = contour.NumberOfContourPoints
        self.points = np.reshape(contour.ContourData, newshape=(-1, 3))

        # Check all points have the same Z coordinate
        z_coords = self.points[:, 2]
        if not np.all(z_coords == z_coords[0]):
            raise ValueError("Points do not all have same Z coordinate")

        self.image_id = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
        self.name = name
        self.color = color

    def plot(self, ax: plt.axis):
        ax.scatter(self.points[:, 0], self.points[:, 1],
                   facecolor=np.array(self.color) / 255.0, label=self.name)
