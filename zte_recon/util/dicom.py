import SimpleITK as sitk
import shutil
import numpy as np
import os
from pydicom import uid

def write_to_dicoms(arr, output_directory, max_val=None,
                    tr=None, flip=None, res=1, save_zip=True, 
                    series_num=None, orientation='ax'):
    '''
    Input Arr needs to be 3D (x,y,z)
    Dicoms saved will be [192, -y, x] and slices will be labeled with input orientation for MPR
    Dicoms will be slices along arr first dimension by default, else transpose

    ZTE Data Obj can be provided optionally to save params
    '''

 # Generate series subdirectory if specified
    if series_num is not None:
        output_directory = output_directory + '_Series' + str(series_num) + '/'
    # If output path doesn't exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Transpose to slice along last dim
    arrN = abs(arr)
    if max_val is None:
        max_val = np.iinfo(np.uint16).max
    else:
        assert max_val <= np.iinfo(np.uint16).max
    arrN = (arrN / np.max(arrN)) * max_val
    arrN = arrN.astype(np.uint16)

    # Convert NumPy array to SimpleITK image
    # SimpleITK expects (X, Y, Z) order, so we transpose the array
    itk_image = sitk.GetImageFromArray(arrN)
    # Set spacing in mm
    itk_image.SetSpacing((res, res, res))  # (x, y, z)
    
    # Use consistent UIDs for the whole series
    study_uid = uid.generate_uid()
    series_uid = uid.generate_uid()

    # Set optional provided params
    if tr is not None:
        itk_image.SetMetaData("0018|0080", str(tr)) 
    if flip is not None:
        itk_image.SetMetaData("0018|1314", str(flip)) 

    # Write the image as a DICOM series
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    if orientation=='ax':
        orient_vals = ['1', '0', '0', '0', '1', '0']

    # Loop through slices to write individual DICOM files
    for i in range(itk_image.GetDepth()):
        slice_image = itk_image[:, :, i] # slice along first dim
        
        # Set metadata for each slice
        slice_image.SetMetaData("0008|0060", "MR")  # Modality (e.g., CT, MR, OT = other)
        slice_image.SetMetaData("0020|0013", str(i + 1)) # Instance Number (slice number)
        slice_image.SetMetaData("0020|000D", study_uid)         # StudyInstanceUID
        slice_image.SetMetaData("0020|000E", series_uid)        # SeriesInstanceUID
        slice_image.SetMetaData("0008|0018", uid.generate_uid())  # SOPInstanceUID
        slice_image.SetMetaData("0020|0032", '\\'.join(map(str, 
                                itk_image.TransformIndexToPhysicalPoint((0, 0, i))))) # ImagePosition
        slice_image.SetMetaData("0020|0037", '\\'.join(orient_vals)) # ImageOrientation
        slice_image.SetMetaData("0028|1050", str(max_val // 2))   # Window Center
        slice_image.SetMetaData("0028|1051", str(max_val))  # Window Width

        
        filename = os.path.join(output_directory, f"image_{i:04d}.dcm")
        
        # Write dicom file
        writer.SetFileName(filename)
        writer.Execute(slice_image)

    if save_zip:
        # Save zip in current directory for download
        shutil.make_archive('dicoms_for_download', 'zip', output_directory)
        print('Zipped folder for download saved to ./dicom_for_download.zip')

    print(f"DICOM series successfully written to: {output_directory}")


def write_timeseries_to_dicoms(arr, output_directory, slice_dim=0, 
                    tr=None, flip=None, res=1, save_zip=True):
    
    '''
    Input arr dims [time, x, y, z]

    '''
    # New series for each time index in first dim
    for t in range(len(arr)):
        write_to_dicoms(arr[t], output_directory, slice_dim, tr, flip, res, save_zip=False, series_num=t)

    # Create zip after all subdirectories made
    if save_zip:
        # Save zip in current directory for download
        shutil.make_archive('dicoms_for_download', 'zip', output_directory)
        print('Zipped folder for download saved to ./dicom_for_download.zip')


