"""A class for manipulating label maps contained in RTSTRUCT files.
  Contributors include:
  David Edmunds
  Nadya Shusharina
  Thomas Bortfeld
  Licensed under the MIT (LICENSE.txt) license.
"""

import copy
import math
from typing import List, Dict, Tuple

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import skimage.draw
from bidict import bidict
from matplotlib.patches import Patch

from ctv_delineation import dcm_image, rtstruct


class LabelMap:
    """An array of unique integer labels, each corresponding to a different
    structure type.

    Attributes:
        array (np.ndarray): The underlying Numpy array of integers.
        names (bidict): A dictionary mapping integers to structure names.
    """

    def __init__(self, array, names):
        self.array = array
        self.names = names

    @classmethod
    def from_dicom(cls,
                   image: dcm_image.DICOMImage,
                   rtstruct_file: rtstruct.RTStructFile):
        """Creates a LabelMap from a DICOM image and an RTSTRUCT DICOM file.

        Args:
            image: The DICOM image object.
            rtstruct_file: The RTStructFile object.
        """
        return cls(cls._get_label_array(image, rtstruct_file),
                   cls._get_name_dict(rtstruct_file))

    @classmethod
    def from_mha(cls,
                 mha_file: str,
                 structure_file: str):
        """Creates a LabelMap from a MHA file and structure list.

        Args:
            mha_file: File name of MHA file containing structures.
            structure_file: File name of structure list.
        """
        return cls(cls._get_mha_label_array(fn_mha),
                   cls._get_mha_name_dict(fn_structlist))

    @classmethod
    def _get_label_array(cls,
                         image: dcm_image.DICOMImage,
                         rtstruct_file: rtstruct.RTStructFile) -> np.ndarray:
        """Extracts the label array from the RTSTRUCT file.

        Knowledge of the DICOM image that the RTSTRUCT file is associated
        with is necessary to correctly calculate the voxel grid coordinates.

        Args:
            image: The DICOM image object.
            rtstruct_file: The RTStructFile object.

        Returns:
            A numpy array of integer labels.
        """
        label_array = np.zeros(shape=image.image.shape, dtype=int)
        for structure in rtstruct_file.structures:
            for contour in structure.contours:
                voxel_indices = ((contour.points - image.origin) /
                                 image.pixel_spacing)
                voxel_indices = voxel_indices.astype(int)
                rr, cc = skimage.draw.polygon(voxel_indices[:, 1],
                                              voxel_indices[:, 0])
                z_index = voxel_indices[:, 2][0]
                z_vec = np.ones_like(rr) * z_index
                label_array[rr, cc, z_vec] = structure.number
        return label_array

    @classmethod
    def _get_mha_label_array(cls, fn_mha: str) -> np.ndarray:
        """Extracts the label array from the MHA file.
 
       Args:
            fn_mha: The mha file name

        Returns:
            A numpy array of integer labels.
        """
        image_reader = vtk.vtkMetaImageReader()
        image_reader.SetFileName(fn_mha)
        image_reader.Update()
        image = image_reader.GetOutput()

        rows, cols, _ = image.GetDimensions()
        a_raw = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())

        # Convert from VTK into numpy array of type int, proper index layout
        nx,ny,nz = image.GetDimensions()	
        label_array = np.zeros(nx*ny*nz, dtype=int)
        for i in range(len(a_raw)):
            label_array[i] = a_raw[i]
        label_array = label_array.reshape( (nx,ny,nz), order='F')

        return label_array

    @classmethod
    def _get_name_dict(cls,
                       rtstruct_file: rtstruct.RTStructFile) -> bidict:
        """Gets a dictionary which maps structure ID numbers to names.

        Args:
            rtstruct_file: The RTStructFile object.

        Returns:
            bidict[int, str]: Bi-directional dict between ID numbers and
                structure names.
        """
        numbers = [int(s.number) for s in rtstruct_file.structures]
        names = [s.name for s in rtstruct_file.structures]
        name_dict = bidict()
        for number, name in zip(numbers, names):
            name_dict[number] = name
        name_dict[0] = 'Background'
        return name_dict
    
    @classmethod
    def _get_mha_name_dict(cls,
                       fn_structlist) -> bidict:
        """Gets a dictionary which maps structure ID numbers to names.

        Args:
            fn_structlist: file name of input list

        Returns:
            bidict[int, str]: Bi-directional dict between ID numbers and
                structure names.
        """
        lines = np.genfromtxt(fn_structlist, delimiter=",", dtype=str)
        name_dict = bidict()
        for i in range(len(lines)):
           name_dict[int(lines[i][0])] = lines[i][1]
           numbers = lines[i][0]
           names = lines[i][1]
  
        return name_dict

    def get_voxels_for_structure(self, name: str) -> List:
        """Get a list of all voxels which belong to a given structure.

        Args:
            name (str): The name of the structure to find.

        Returns:
            A list of (i, j, k) tuples representing the indices of all voxels
            belonging to the structure.
        """
        index = self.names.inverse[name]
        return [tuple(node) for node in np.argwhere(self.array == index)]

    def get_slices_with_structure(self, name: str) -> np.ndarray:
        """Returns a list of axial slices that contain a given structure.

        Args:
            name (str): The name of the structure to find.

        Returns:
            A list of slice indexes for slices which contain the structure.
        """
        index = self.names.inverse[name]
        nodes = [tuple(node) for node in np.argwhere(self.array == index)]
        nodes = np.row_stack(nodes)
        slices = nodes[:, 2]
        slices = np.unique(slices)
        return slices

    def get_slices_close_to_gtv(self, threshold: int) -> Tuple:
        """Returns slices close to the GTV.

        Returns the index of the min and max axial slices containing GTV,
        plus a threshold amount.

        Args:
            threshold (int): The amount below and above

        Returns:
            Min and max slice indices.
        """
        slices = self.get_slices_with_structure('gtv')
        return np.min(slices) - threshold, np.max(slices) + threshold

    def crop(self, row_min, row_max, col_min, col_max, slice_min, slice_max):
        """Returns a cropped version of the original LabelMap.

        Args:
            row_min (int): Min row index to keep.
            row_max (int): Max row index to keep.
            col_min (int): Min col index to keep.
            col_max (int): Max col index to keep
            slice_min (int): Min slice index to keep.
            slice_max (int): Max slice index to keep.

        Returns:
            A new, cropped LabelMap object.
        """
        return LabelMap(self.array[row_min: row_max + 1,
                                   col_min: col_max + 1,
                                   slice_min: slice_max + 1],
                        self.names)


def get_graph(label_map: LabelMap,
              pixel_spacing: np.ndarray,
              impermeable_structure_names: List[str],
              permeable_structures_dict: Dict[str, float]) -> nx.Graph:
    """Converts a label map to a NetworkX Graph object.

    Args:
        label_map (LabelMap): The LabelMap to convert.
        pixel_spacing (np.ndarray): A length-3 array of pixel spacings. This
            is necessary to calculate the correct distance weightings for edges
            in the graph.
        impermeable_structure_names (List[str]): A list of structure
            names to be treated as impermeable by the graph.
        permeable_structures_dict (Dict[str, float]): A dict mapping
            structure names to resistance factors, which will be used to
            modify the edge weights for permeable structures.
    """
    G = nx.Graph()

    r, c, s = pixel_spacing

    # Simplifying assumption: r == c, otherwise we have to calculate different
    # diagonal distances for different directions
    assert r == c

    axial_diagonal = math.sqrt(r ** 2 + c ** 2)
    nesw_vert_diag = math.sqrt(s ** 2 + c ** 2)  # North East South West
    ne_nw_se_sw_vert_diag = math.sqrt(s ** 2 + axial_diagonal ** 2)

    nrows, ncols, nslices = label_map.array.shape

    # Create 3D grid of nodes, with no edges yet
    G.add_nodes_from(
        (i, j, k) for i in range(nrows) for j in range(ncols) for k in
        range(nslices))

    # Add down connections
    G.add_edges_from(
        [((i, j, k), (i + 1, j, k)) for i in range(nrows - 1) for j in
         range(ncols) for k in range(nslices)], weight=r)

    # Add right connections
    G.add_edges_from(
        [((i, j, k), (i, j + 1, k)) for i in range(nrows) for j in
         range(ncols - 1) for k in range(nslices)], weight=c)

    # Add down-right diagonal connections
    G.add_edges_from(
        [((i, j, k), (i + 1, j + 1, k)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices)], weight=axial_diagonal)

    # Add down-left diagonal connections
    G.add_edges_from(
        [((i, j + 1, k), (i + 1, j, k)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices)], weight=axial_diagonal)

    # Add vertical connections between slices
    G.add_edges_from(
        [((i, j, k), (i, j, k + 1)) for i in range(nrows) for j in range(ncols)
         for k in range(nslices - 1)], weight=s)

    # Add down connections between slices
    G.add_edges_from(
        [((i, j, k), (i + 1, j, k + 1)) for i in range(nrows - 1) for j in
         range(ncols) for k in range(nslices - 1)], weight=nesw_vert_diag)

    # Add up connections between slices
    G.add_edges_from(
        [((i + 1, j, k), (i, j, k + 1)) for i in range(nrows - 1) for j in
         range(ncols) for k in range(nslices - 1)], weight=nesw_vert_diag)

    # Add left connections between slices
    G.add_edges_from(
        [((i, j + 1, k), (i, j, k + 1)) for i in range(nrows) for j in
         range(ncols - 1) for k in range(nslices - 1)], weight=nesw_vert_diag)

    # Add right connections between slices
    G.add_edges_from(
        [((i, j, k), (i, j + 1, k + 1)) for i in range(nrows) for j in
         range(ncols - 1) for k in range(nslices - 1)], weight=nesw_vert_diag)

    # Add down-right connections between slices
    G.add_edges_from(
        [((i, j, k), (i + 1, j + 1, k + 1)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices - 1)],
        weight=ne_nw_se_sw_vert_diag)

    # Add up-right connections between slices
    G.add_edges_from(
        [((i + 1, j, k), (i, j + 1, k + 1)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices - 1)],
        weight=ne_nw_se_sw_vert_diag)

    # Add up-left connections between slices
    G.add_edges_from(
        [((i + 1, j + 1, k), (i, j, k + 1)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices - 1)],
        weight=ne_nw_se_sw_vert_diag)

    # Add down-left connections between slices
    G.add_edges_from(
        [((i, j + 1, k), (i + 1, j, k + 1)) for i in range(nrows - 1) for j in
         range(ncols - 1) for k in range(nslices - 1)],
        weight=ne_nw_se_sw_vert_diag)

    # Delete nodes corresponding to impermeable structures
    for name in impermeable_structure_names:
        nodes = label_map.get_voxels_for_structure(name)
        G.remove_nodes_from(nodes)

    # Alter the edge weights for semi-permeable structures
    for name, resistance in permeable_structures_dict.items():
        nodes = label_map.get_voxels_for_structure(name)
        edges = G.edges(nodes)
        for src, dst in edges:
            G[src][dst]['weight'] *= resistance

    return G


def ctv_expansion(G, label_map: LabelMap, cutoff: int):
    """Runs the CTV expansion algorithm to grow the CTV.

    Parameters:
        G (nx.Graph): The graph representation of the label map to expand.
        label_map: The label map to expand.
        cutoff: Maximum distance that will be checked for CTV expansion.
    """
    gtv_nodes = label_map.get_voxels_for_structure('gtv')
    all_length = nx.multi_source_dijkstra_path_length(G,
                                                      gtv_nodes,
                                                      cutoff=cutoff)
    ctv_indices = list(all_length.keys())

    ctv_indices = set(ctv_indices).difference(set(gtv_nodes))

    label_map_with_ctv = copy.deepcopy(label_map)
    label_map_with_ctv.names[999] = 'CTV'
    for index in ctv_indices:
        label_map_with_ctv.array[index] = all_length[index]
    return label_map_with_ctv


def plot(label_map: LabelMap, slice_index: int, cmap_name='tab20'):
    """Plots a single LabelMap slice.

    Args:
        label_map (LabelMap): The LabelMap to plot.
        slice_index (int): Index of the axial slice to plot.
        cmap_name (str): Name of the Matplotlib colormap to use.
    """

    plt.figure(figsize=(10, 10))
    image = label_map.array[:, :, slice_index]
    label_indices = np.unique(image)
    label_names = [label_map.names[i] for i in label_indices]

    cmap = matplotlib.cm.get_cmap(cmap_name)

    norm = matplotlib.colors.BoundaryNorm(boundaries=label_indices,
                                          ncolors=len(label_indices))
    plt.imshow(image, cmap=cmap, norm=norm)

    handles = [Patch(facecolor=cmap(norm(index)), label=name) for
               index, name in zip(label_indices, label_names)]
    plt.legend(handles=handles)

def plot_with_colors(label_map: LabelMap, slice_index: int, color_dict: Dict):
    """Plot a single LabelMap slice with user-defined colors.

    Args:
        label_map (LabelMap): The LabelMap to plot.
        slice_index (int): Index of the axial slice to plot.
        color_dict: A dictionary of (structure_name, color) pairs.
    """
    plt.figure(figsize=(10, 10))
    image = label_map.array[:, :, slice_index]
    nrows, ncols = image.shape
    colored = np.zeros(shape=(nrows, ncols, 3))
    handles = []
    for structure_name, color in color_dict.items():
        structure_id = label_map.names.inverse[structure_name]
        colored[np.where(image == structure_id)] = color
        handles.append(Patch(facecolor=color, label=structure_name))
    plt.imshow(colored)
    plt.legend(handles=handles)

def plot_completemap(label_map: LabelMap, slice_index: int, cmap_name='tab20'):
    """Plots a single LabelMap slice.

    Args:
        label_map (LabelMap): The LabelMap to plot.
        slice_index (int): Index of the axial slice to plot.
        cmap_name (str): Name of the Matplotlib colormap to use.
    """

    plt.figure(figsize=(10, 10))

    image = label_map.array[:, :, slice_index]
    label_indices = np.unique(image)
#    print(label_indices)
#    label_names = [label_map.names[i] for i in label_indices]

    cmap = matplotlib.cm.get_cmap(cmap_name)

    norm = matplotlib.colors.BoundaryNorm(boundaries=label_indices,
                                          ncolors=len(label_indices))
    plt.imshow(image, cmap=cmap)#, norm=norm)

    #handles = [Patch(facecolor=cmap(norm(index)), label=str(index)) for
    #           index in label_indices ]
    #plt.legend(handles=handles)

def write_mha_float( label_map: LabelMap, fn):
    
    #cast to float
    (nx,ny,nz) = label_map.array.shape
    label_array = copy.deepcopy(label_map.array)
    float_array = np.zeros(nx*ny*nz, dtype='float32')
    lm_raw = np.ravel(label_array, order='F')
    for i in range(len(lm_raw)):
            float_array[i] = lm_raw[i]
    print(nx, ny, nz, len(float_array))
    print(float_array.itemsize)
    f = open(fn,'w')

    f.write('ObjectType = Image\n')
    f.write('NDims = 3\n')
    f.write('BinaryData = True\n')
    f.write('BinaryDataByteOrderMSB = False\n')
    f.write('CompressedData = False\n')
    f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
    f.write('Offset = 0 0 0\n')
    f.write('CenterOfRotation = 0 0 0\n')
    f.write('AnatomicalOrientation = RAI\n')
    f.write('ElementSpacing = 1 1 3\n')
    f.write('DimSize = %d %d %d\n' % (nx, ny, nz))
    f.write('ElementType = MET_FLOAT\n')
    f.write('ElementDataFile = LOCAL\n')

    float_array.tofile(f,sep="")
    f.close()

    return
    VTK_data = numpy_support.numpy_to_vtk(num_array=float_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(nx, ny, nz)
    imageData.SetOrigin(0.0, 0.0, 0.0)
    imageData.SetSpacing(1, 1, 3)
    #imageData.SetPointDataActiveScalarInfo( VTK_FLOAT, 1) ????
    imageData.GetPointData().SetScalars(VTK_data)
    wr = vtk.vtkMetaImageWriter()
    wr.SetFileName("dd.mha")
    wr.SetInputData(imageData)
    wr.Write()

def write_mha_uchar( label_map: LabelMap, fn):
    #cast to float
    (nx,ny,nz) = label_map.array.shape
    label_array = copy.deepcopy(label_map.array)
    float_array = np.zeros(nx*ny*nz, dtype='uint8')
    lm_raw = np.ravel(label_array, order='F')
    for i in range(len(lm_raw)):
            float_array[i] = lm_raw[i]
    print(nx, ny, nz, len(float_array))
    print(float_array.itemsize)
    f = open(fn,'w')

    f.write('ObjectType = Image\n')
    f.write('NDims = 3\n')
    f.write('BinaryData = True\n')
    f.write('BinaryDataByteOrderMSB = False\n')
    f.write('CompressedData = False\n')
    f.write('TransformMatrix = 1 0 0 0 1 0 0 0 1\n')
    f.write('Offset = 0 0 0\n')
    f.write('CenterOfRotation = 0 0 0\n')
    f.write('AnatomicalOrientation = RAI\n')
    f.write('ElementSpacing = 1 1 3\n')
    f.write('DimSize = %d %d %d\n' % (nx, ny, nz))
    f.write('ElementType = MET_UCHAR\n')
    f.write('ElementDataFile = LOCAL\n')

    float_array.tofile(f,sep="")
    f.close()

    return
    VTK_data = numpy_support.numpy_to_vtk(num_array=float_array.ravel(), deep=True, array_type=vtk.VTK_UCHAR)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(nx, ny, nz)
    imageData.SetOrigin(0.0, 0.0, 0.0)
    imageData.SetSpacing(1, 1, 3)
    #imageData.SetPointDataActiveScalarInfo( VTK_FLOAT, 1) ????
    imageData.GetPointData().SetScalars(VTK_data)
    wr = vtk.vtkMetaImageWriter()
    wr.SetFileName("dd.mha")
    wr.SetInputData(imageData)
    wr.Write()

def paste_into_background( LM_bg: LabelMap, LM_fg: LabelMap, offsx, offsy, offsz):
    ### paste LM_fg image onto LM_bg image at offset offsx,offsy,offsz voxels
    ### no checking of boundaries etc
    print('paste me!')
    print(offsx, offsy, offsz)
    print(LM_bg.array.shape)
    print(LM_fg.array.shape)
    output = copy.deepcopy(LM_bg)
    for z in range(offsz, offsz+LM_fg.array.shape[2]):
        for y in range(offsy, offsy+LM_fg.array.shape[1]):
            for x in range(offsx, offsx + LM_fg.array.shape[0]):
                output.array[x][y][z] = LM_fg.array[x-offsx][y-offsy][z-offsz]
    return output

