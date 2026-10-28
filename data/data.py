import torch
from pathlib import Path
from.preprocess import *


# # 图像读取
# def img_read(
#         img_path,
#         target_spacing=(1, 1, 1), # 采样间距 (z, y, x), mm
#         orient="LPS",
#         interpolate_mode = "trilinear"  # trilinear | nearest
# ):
#     # 图像读取
#     img = sitk.ReadImage(str(img_path))
#     img_size = img.GetSize() # (x, y, z)
#     # 坐标系处理
#     native_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
#         img.GetDirection()
#     )
#     orient_filter = sitk.DICOMOrientImageFilter()
#     orient_filter.SetDesiredCoordinateOrientation(orient)
#     img = orient_filter.Execute(img)
#     # 尺度处理
#     native_spacing = list(img.GetSpacing())
#     arr = sitk.GetArrayFromImage(img)
#     # arr shape is (D, H, W), native_spacing is (x, y, z)
#     z, y, x = native_spacing[2], native_spacing[1], native_spacing[0]
#     target_size = (
#         int(arr.shape[0] * z / target_spacing[0]),  # D
#         int(arr.shape[1] * y / target_spacing[1]),  # H
#         int(arr.shape[2] * x / target_spacing[2]),  # W
#     )
#     # Resample via interpolation
#     arr_t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
#     arr_t = torch.nn.functional.interpolate(arr_t, size=target_size, mode=interpolate_mode).squeeze()
    
#     return arr_t.numpy()


# ============================================== 加载数据 ==============================================
def load_nifti_file(fpath):
    """
    Load a NIfTI file using SimpleITK.
    
    Args:
        fpath (str or Path): Path to NIfTI file (.nii or .nii.gz).
    
    Returns:
        SimpleITK.Image or None: Loaded image, or None if loading fails
            or image has multiple components (e.g., RGB).
    """
    try:
        img_sitk = sitk.ReadImage(str(fpath))
        
        # Validate image format - reject multi-component images (e.g., RGB)
        if img_sitk.GetNumberOfComponentsPerPixel() > 1:
            print(f"Warning: Multi-component image detected. Skipping {fpath}")
            return None
        
        return img_sitk
    except Exception as e:
        print(f"Error loading NIfTI file {fpath}: {e}")
        return None

def load_dicom_file(path):
    """
    Load DICOM data (single file or series from directory) using SimpleITK.
    
    Args:
        path (str or Path): Path to either:
            - A directory containing a DICOM series (multiple .dcm files)
            - A single DICOM file
    
    Returns:
        SimpleITK.Image or None: Loaded image, or None if loading fails,
            no DICOM files found, or image has multiple components.
    """
    path = Path(path)
    
    try:
        # Read DICOM series or single file
        if path.is_dir():
            # Read DICOM series from directory
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            
            if len(dicom_names) == 0:
                print(f"Warning: No DICOM files found in {path}")
                return None
            
            reader.SetFileNames(dicom_names)
            img_sitk = reader.Execute()
        else:
            # Read single DICOM file
            img_sitk = sitk.ReadImage(str(path))
        
        # Validate image format - reject multi-component images (e.g., RGB)
        if img_sitk.GetNumberOfComponentsPerPixel() > 1:
            print(f"Warning: Multi-component image detected. Skipping {path}")
            return None
        
        return img_sitk
    except Exception as e:
        print(f"Error loading DICOM file/series {path}: {e}")
        return None

# ============================================== 载入图像 ==============================================
def load_image(path, preprocess=True):
    """
    Load and preprocess a medical image with automatic format detection.
    
    This is the main entry point for loading medical images. It automatically
    detects whether the input is NIfTI or DICOM format and applies the
    appropriate loading method, followed by optional standardized preprocessing.
    
    Format Detection Logic:
    - Files ending in .nii or .nii.gz → NIfTI
    - Files ending in .dcm or .dicom → DICOM
    - Directories → Assumed to contain DICOM series
    - Other extensions → Attempts generic SimpleITK loading (may fail)
    
    Preprocessing Pipeline (if preprocess=True):
    1. Reorients to standard RPI orientation
    2. Determines slice dimension based on spacing/size heuristics
    3. Resamples to anisotropic resolution (1x1x4mm)
    4. Crops to dimensions divisible by 16 (in-plane) and 4 (through-plane)
    
    Args:
        path (str or Path): Path to medical image file or directory containing
            DICOM series. Supported formats:
            - NIfTI: .nii, .nii.gz
            - DICOM: .dcm, .dicom, or directory with DICOM series
        preprocess (bool, optional): Whether to apply standardized preprocessing
            (reorientation, resampling, cropping). Defaults to True.
    
    Returns:
        SimpleITK.Image or None: Loaded (and optionally preprocessed) 3D image,
            or None if loading fails or image format is unsupported.
    
    Raises:
        FileNotFoundError: If the specified path doesn't exist.
    
    Examples:
        >>> # Load and preprocess a NIfTI file
        >>> img = load_image('brain_scan.nii.gz')
        >>> 
        >>> # Load DICOM without preprocessing
        >>> img = load_image('dicom_series/', preprocess=False)
        >>> 
        >>> # Load and convert to numpy
        >>> img = load_image('scan.nii.gz')
        >>> if img is not None:
        ...     arr = sitk.GetArrayFromImage(img)
        ...     print(f"Loaded volume: {arr.shape}, dtype: {arr.dtype}")
    
    Notes:
        - Multi-component images (e.g., RGB) are not supported and will return None
        - DICOM series should be in a single directory with consistent orientation
        - Preprocessing uses BSpline interpolation for high-quality resampling
        - Target spacing: 1x1x4mm (in-plane × through-plane)
    """
    path = Path(path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Determine format and load image
    img_sitk = None
    
    if path.is_dir():
        # Directory: assume DICOM series
        print(f"Loading DICOM series from directory: {path}")
        img_sitk = load_dicom_file(path)
    else:
        # File: detect format by extension
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.gz']:
            # Handle .nii.gz
            if path.name.endswith('.nii.gz'):
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            elif suffix == '.nii':
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            else:
                # .gz but not .nii.gz, might be compressed NIfTI
                print(f"Loading file (assuming NIfTI): {path}")
                img_sitk = load_nifti_file(path)
        
        elif suffix in ['.dcm', '.dicom']:
            print(f"Loading DICOM file: {path}")
            img_sitk = load_dicom_file(path)
        
        else:
            # Unknown format: try generic loading
            print(f"Warning: Unknown format '{suffix}'. Attempting generic load: {path}")
            try:
                img_sitk = sitk.ReadImage(str(path))
                if img_sitk.GetNumberOfComponentsPerPixel() > 1:
                    print(f"Warning: Multi-component image detected. Skipping {path}")
                    return None
            except Exception as e:
                print(f"Error loading file {path}: {e}")
                return None
    
    # Check if loading was successful
    if img_sitk is None:
        print(f"Failed to load image from: {path}")
        return None
    
    # Apply preprocessing if requested
    if preprocess:
        print(f"Applying standardized preprocessing...")
        try:
            img_sitk = preprocess_image(img_sitk)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
    
    print(f"Successfully loaded image: shape={img_sitk.GetSize()}, spacing={img_sitk.GetSpacing()}")
    return img_sitk

def prepare_for_inference(img_sitk, mode, z_dim=None):
    """
    Convert a preprocessed SimpleITK image to model-ready numpy arrays.
    
    This function handles the complete array-level preprocessing pipeline:
    1. Converts SimpleITK image to numpy array
    2. Transposes so slice dimension (4mm spacing) is first [D, H, W]
    3. Applies modality-specific preprocessing:
       - CT: Generates brain/blood/bone windowed views + background mask
       - MRI: Applies percentile clipping and normalization + background mask
    4. Validates minimum dimensions (D>=4, H>=16, W>=16)
    
    Args:
        img_sitk (SimpleITK.Image): Preprocessed SimpleITK image (after
            reorientation, resampling, and cropping from io.load_image).
        mode (str): Imaging modality, either 'ct' or 'mri' (case-insensitive).
        z_dim (int, optional): Which dimension (0, 1, or 2) was the slice
            dimension in the original image coordinate system. If None,
            attempts to infer from spacing (dimension with largest spacing).
    
    Returns:
        tuple: (img_arrs, background_mask, view) or None if validation fails
            - img_arrs (list of numpy.ndarray): List of preprocessed arrays.
              For CT: 3 arrays (brain, blood, bone windows)
              For MRI: 1 array (normalized)
              Each array has shape [D, H, W] with values in [0, 1]
            - background_mask (numpy.ndarray): Binary mask [D, H, W] indicating
              foreground regions (True = tissue, False = background)
            - view (int): Indicator of which axis was the slice dimension (0, 1, or 2)
    
    Returns:
        None: If image dimensions are too small (D<4, H<16, or W<16)
    
    Raises:
        ValueError: If mode is not 'ct' or 'mri'
    
    Example:
        >>> from neurovfm.data.io import load_image
        >>> from neurovfm.data.preprocess import prepare_for_inference
        >>> 
        >>> # Load CT scan
        >>> img = load_image('ct_scan.nii.gz')
        >>> img_arrs, mask, view = prepare_for_inference(img, mode='ct')
        >>> 
        >>> # img_arrs contains 3 windowed views
        >>> print(f"Brain window: {img_arrs[0].shape}")
        >>> print(f"Blood window: {img_arrs[1].shape}")
        >>> print(f"Bone window: {img_arrs[2].shape}")
        >>> 
        >>> # Load MRI scan
        >>> img = load_image('mri_scan.nii.gz')
        >>> img_arrs, mask, view = prepare_for_inference(img, mode='mri')
        >>> print(f"Normalized MRI: {img_arrs[0].shape}")
    
    Notes:
        - Input image should already be preprocessed (reoriented, resampled to 1x1x4mm)
        - CT windows used: Brain (W=80, L=40), Blood (W=200, L=80), Bone (W=2800, L=600)
        - MRI normalization: 0.5th-99.5th percentile clipping followed by min-max scaling
        - Background masks help the model focus on relevant anatomy
    """
    mode = mode.lower()
    if mode not in ['ct', 'mri']:
        raise ValueError(f"Mode must be 'ct' or 'mri', got: {mode}")
    
    # Infer z_dim from spacing if not provided
    if z_dim is None:
        spacing = img_sitk.GetSpacing()
        if np.unique(spacing).shape[0] == 1:
            z_dim = 2 # assume axial
        else:
            z_dim = np.argmax(spacing)  # Dimension with largest spacing (4mm)
    
    # Convert SimpleITK image to numpy array
    # GetArrayFromImage returns (z, y, x) order
    img_arr = sitk.GetArrayFromImage(img_sitk)
    
    # Transpose to [D, H, W] where D is the slice dimension
    img_arr, view = transpose_to_dhw(img_arr, z_dim)
    D, H, W = img_arr.shape
    
    # Validate minimum dimensions
    if D < 4 or H < 16 or W < 16:
        print(f"Warning: Image too small (D={D}, H={H}, W={W}). Minimum: D>=4, H>=16, W>=16")
        return None
    
    # Ensure contiguous array in float64
    img_arr = img_arr.astype(np.float64).copy()
    
    # Apply modality-specific preprocessing
    if mode == 'ct':
        # CT: Generate windowed views and background mask
        background_mask = get_background_mask_ct(img_arr.copy())
        
        # Generate three standard CT windows: brain, blood (subdural), bone
        img_arrs = [
            clip_by_window(img_arr, window_width=80, window_level=40),      # Brain
            clip_by_window(img_arr, window_width=200, window_level=80),     # Blood
            clip_by_window(img_arr, window_width=2800, window_level=600)    # Bone
        ]
    
    elif mode == 'mri':
        # MRI: Percentile clipping and normalization
        background_mask = get_background_mask_mri(img_arr.copy())
        
        # Clip to 0.5th-99.5th percentile to remove outliers
        img_arr = np.clip(
            img_arr,
            np.percentile(img_arr, 0.5),
            np.percentile(img_arr, 99.5)
        )
        
        # Min-max normalization to [0, 1]
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
        
        img_arrs = [img_arr]
    
    return img_arrs, background_mask, view


def load_to_tokens(
        img_path,
        modality="mri", # "mri" or "ct"
        patch_size=(4, 16, 16),
        remove_background=True
        ):
    img_sitk = load_image(str(img_path), preprocess=True)
    result = prepare_for_inference(img_sitk, mode=modality)
    img_arrs, background_mask, view = result
    if modality.lower() == "ct":
            for window_idx, img_arr in enumerate(img_arrs):
                # Tokenize with background filtering
                tokens, coords, filtered = tokenize_volume(
                    img_arr,
                    background_mask,
                    patch_size=patch_size,
                    remove_background=remove_background
                )
                
                # Convert to tensors
                tokens_tensor = torch.from_numpy(tokens).float()
                coords_tensor = torch.from_numpy(coords).long()
                
    else:
        img_arr = img_arrs[0]
        tokens, coords, filtered = tokenize_volume(
            img_arr,
            background_mask,
            patch_size=patch_size,
            remove_background=remove_background
        )
        tokens_tensor = torch.from_numpy(tokens).float()
        coords_tensor = torch.from_numpy(coords).long()

    return tokens_tensor, coords_tensor


if __name__ == "__main__":
    modality = "mri"
    img_path = "/hpc/jtia475/datasets/ISLES-2022/ADC/sub-strokecase0140_ses-0001_adc.nii.gz"
    tokens_tensor, coords_tensor = load_to_tokens(img_path, modality=modality, patch_size=(4, 16, 16), remove_background=True)

    print(f"Tokenized volume: {tokens_tensor.shape}, coords: {coords_tensor.shape}")