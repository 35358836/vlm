import numpy as np
from typing import Tuple
import SimpleITK as sitk

def reorient(img_sitk, tgt='RPI'):
    """
    Reorient a medical image to a target anatomical orientation.
    
    This function standardizes the anatomical orientation of medical images,
    which is crucial for consistent processing across different acquisition
    protocols and scanner manufacturers.
    
    Standard orientations:
        - 'RPI': Right-Posterior-Inferior (recommended for Python/NumPy)
        - 'LPS': Left-Posterior-Superior (common in 3D Slicer)
    
    Args:
        img_sitk (SimpleITK.Image): Input SimpleITK image of shape [x, y, z].
        tgt (str, optional): Target orientation string. Each character represents
            an anatomical direction:
            - First char: L (Left) or R (Right)
            - Second char: P (Posterior) or A (Anterior)  
            - Third char: I (Inferior) or S (Superior)
            Defaults to 'RPI'.
    
    Returns:
        SimpleITK.Image: Reoriented image with the same physical space but
            potentially transposed dimensions.
    
    Example:
        >>> img = sitk.ReadImage('scan.nii.gz')
        >>> img_rpi = reorient(img, tgt='RPI')
    """
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(tgt)
    return orienter.Execute(img_sitk)



def compute_spacing(img_sitk):
    """
    Extract the physical spacing (resolution) from a SimpleITK image.
    
    The spacing represents the physical distance between adjacent voxels
    in millimeters for each dimension.
    
    Args:
        img_sitk (SimpleITK.Image): Input SimpleITK image.
    
    Returns:
        numpy.ndarray: Array of shape (3,) containing spacing values [x, y, z]
            in millimeters for each spatial dimension.
    
    Example:
        >>> img = sitk.ReadImage('scan.nii.gz')
        >>> spacing = compute_spacing(img)
        >>> print(f"Voxel spacing: {spacing} mm")
        Voxel spacing: [0.5 0.5 3.0] mm
    """
    spacing_sitk = img_sitk.GetSpacing()
    return np.array(spacing_sitk, dtype=float)


def preprocess_image(img_sitk):
    """
    Apply standardized preprocessing to a medical image.
    
    This function performs the complete preprocessing pipeline:
    1. Reorients to standard RPI orientation
    2. Determines slice dimension based on spacing/size heuristics
    3. Resamples to anisotropic resolution (1x1x4mm)
    4. Crops to dimensions divisible by 16 (in-plane) and 4 (through-plane)
    
    The slice dimension (z_dim) is automatically detected using the following logic:
    - If all spacings are equal: defaults to dimension 0
    - If sizes are all different: uses dimension with largest spacing
    - Otherwise: uses dimension with unique size (typically slice direction)
    
    Target spacing after resampling:
    - In-plane dimensions: 1.0 mm
    - Through-plane (slice) dimension: 4.0 mm
    
    Args:
        img_sitk (SimpleITK.Image): Input medical image (NIfTI or DICOM).
    
    Returns:
        SimpleITK.Image: Preprocessed 3D image with standardized spacing
            and dimensions suitable for model inference.
    
    Notes:
        - Cropping ensures dimensions are compatible with patch-based processing
        - BSpline interpolation is used for high-quality resampling
    """
    # Reorient to standard RPI coordinate system
    # RPI = Right-Posterior-Inferior (standard for medical image processing)
    img_sitk = reorient(img_sitk, tgt='RPI')  # (x, y, z)
    
    # Extract physical spacing information
    spacing_sitk = compute_spacing(img_sitk)  # -> np.array([x, y, z])
    
    original_spacing = spacing_sitk
    original_size = img_sitk.GetSize()
    
    # Determine slice dimension (z_dim) using heuristics
    # This identifies which dimension represents the slice/through-plane axis
    if len(set(list(original_spacing))) == 1:
        # All spacings equal - default to dimension 2
        z_dim = 2
    else:
        counts = np.bincount(original_size)
        if (counts == 1).all():
            # All sizes are unique - use dimension with largest spacing
            z_dim = np.argmax(original_spacing)
        else:
            # Use dimension with unique size (typically slice direction)
            z_dim = np.where(original_size == np.where(counts == 1)[0][0])[0][0]
    
    # Define target anisotropic spacing: 1x1x4 mm
    # In-plane: 1mm × 1mm for high resolution
    # Through-plane: 4mm for typical slice thickness
    target_spacing = [1, 1, 1]
    target_spacing[z_dim] = 4
    
    # Calculate new size based on spacing ratio
    new_size = [
        int(original_size[i] * (original_spacing[i] / target_spacing[i])) 
        for i in range(len(original_size))
    ]
    
    # Configure resampler with BSpline interpolation for smooth results
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Execute resampling
    resized_img_sitk = resampler.Execute(img_sitk)
    
    # Crop to dimensions divisible by patch sizes
    # In-plane: divisible by 16 for patch-based processing
    # Through-plane: divisible by 4 for slice-based processing
    start_index, crop_size = [], []
    for idx in range(3):
        if idx == z_dim:
            # Slice dimension: make divisible by 4
            start_index.append((new_size[idx] % 4) // 2)
            crop_size.append((new_size[idx] // 4) * 4)
        else:
            # In-plane dimensions: make divisible by 16
            start_index.append((new_size[idx] % 16) // 2)
            crop_size.append((new_size[idx] // 16) * 16)
    
    # Extract cropped region (centered crop)
    resized_img_sitk = sitk.Extract(resized_img_sitk, crop_size, start_index)
    
    return resized_img_sitk



def transpose_to_dhw(img_arr, z_dim):
    """
    Transpose image array so the slice dimension (resampled 4mm axis) is first.
    
    After resampling, the slice dimension (with 4mm spacing) needs to be
    the first dimension [D, H, W] for proper model processing.
    
    Args:
        img_arr (numpy.ndarray): Image array from SimpleITK (z, y, x) order.
        z_dim (int): Which dimension (0, 1, or 2) is the slice dimension
            in the original SimpleITK coordinate system.
    
    Returns:
        tuple: (transposed_array, view_indicator)
            - transposed_array: Array with shape [D, H, W] where D is slice dim
            - view_indicator: Integer (0, 1, or 2) indicating original slice axis
    
    Notes:
        - SimpleITK.GetArrayFromImage returns array in (z, y, x) order
        - We need to transpose so the 4mm-spaced dimension is first
    """
    # SimpleITK returns array in (z, y, x) order
    # z_dim indicates which axis in (x, y, z) space is the slice dimension
    # We need to map this to the array indexing
    
    # Convert z_dim from (x,y,z) to array (z,y,x) indexing
    view = 2 - z_dim
    
    # Transpose to put slice dimension first
    if view == 0:
        # Already in correct order [D, H, W]
        return img_arr, view
    elif view == 1:
        # Need to swap: (z, D, x) -> (D, z, x)
        return np.transpose(img_arr, (1, 0, 2)), view
    elif view == 2:
        # Need to swap: (z, y, D) -> (D, y, z)
        return np.transpose(img_arr, (2, 1, 0)), view
    else:
        # Default: no transpose
        return img_arr, view

    

def get_background_mask_mri(img_arr):
    """
    Generate a background mask for MRI images.
    
    Identifies background regions in MRI images based on intensity statistics.
    Background in MRI is typically low intensity noise outside the anatomy.
    
    Args:
        img_arr (numpy.ndarray): MRI image array of shape [D, H, W].
    
    Returns:
        numpy.ndarray: Binary mask of shape [D, H, W] where True indicates
            foreground (anatomy) and False indicates background.
    
    Notes:
        - MRI background is typically near-zero intensity
        - Uses percentile-based thresholding to separate signal from noise
    """
    # MRI-specific background detection
    # Background is typically low intensity
    threshold = np.percentile(img_arr, 10)
    mask = img_arr > threshold
    return mask



def get_background_mask_ct(img_arr):
    """
    Generate a background mask for CT images.
    
    Identifies background regions in CT images based on intensity thresholds.
    Background voxels are typically air or regions outside the patient.
    
    Args:
        img_arr (numpy.ndarray): CT image array of shape [D, H, W].
    
    Returns:
        numpy.ndarray: Binary mask of shape [D, H, W] where True indicates
            foreground (patient tissue) and False indicates background.
    
    Notes:
        - CT values are in Hounsfield Units (HU)
        - Air is approximately -1000 HU
        - This function identifies regions likely to be background
    """
    # CT-specific background detection
    # Typically, background in CT is very low intensity (air ~ -1000 HU)
    mask = img_arr > -500  # Threshold for separating air from tissue
    return mask



def clip_by_window(img_arr, window_width, window_level):
    """
    Apply CT windowing to enhance specific tissue contrast.
    
    CT windowing (also called window/level adjustment) enhances visualization
    of specific tissue types by mapping a range of Hounsfield Units to the
    full display range [0, 1].
    
    Args:
        img_arr (numpy.ndarray): CT image array in Hounsfield Units.
        window_width (int or float): Width of the window (range of HU values).
        window_level (int or float): Center of the window (middle HU value).
    
    Returns:
        numpy.ndarray: Windowed image normalized to [0, 1].
    
    Example:
        >>> # Brain window (good for brain tissue)
        >>> brain = clip_by_window(ct_array, window_width=80, window_level=40)
        >>> 
        >>> # Bone window (good for skeletal structures)
        >>> bone = clip_by_window(ct_array, window_width=2800, window_level=600)
    
    Notes:
        Common CT windows:
        - Brain: W=80, L=40
        - Subdural: W=200, L=80 (blood)
        - Bone: W=2800, L=600
        - Lung: W=1500, L=-600
        - Abdomen: W=400, L=50
    """
    # Calculate window boundaries
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    
    # Clip and normalize to [0, 1]
    windowed = np.clip(img_arr, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8)
    
    return windowed



def tokenize_volume(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    patch_size: Tuple[int, int, int] = (4, 16, 16),
    remove_background: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize a 3D volume into patches for model inference.
    
    Converts a 3D image volume [D, H, W] into a sequence of flattened patches.
    Each patch is flattened into a 1D vector, creating a tokens × features format.
    Also generates a background/foreground mask for each token.
    
    Args:
        img_arr (numpy.ndarray): Image array of shape [D, H, W] with values in [0, 1].
        mask_arr (numpy.ndarray): Binary mask of shape [D, H, W] (True = foreground).
        patch_size (tuple): Size of patches (depth, height, width). 
            Defaults to (4, 16, 16) for 1024-D tokens.
        remove_background (bool): If True, physically remove background tokens.
            If False, return all tokens with a filtered mask. Defaults to False.
    
    Returns:
        tuple: (tokens, coords, filtered)
            - tokens (numpy.ndarray): Tokenized patches [N, patch_features] or [N_fg, patch_features]
            - coords (numpy.ndarray): 3D coordinates [N, 3] or [N_fg, 3] for each token
            - filtered (numpy.ndarray): Background mask [N] where 1=background, 0=foreground
                                        (empty if remove_background=True)
    
    Example:
        >>> img_arr = np.random.rand(64, 192, 192)  # D, H, W
        >>> mask_arr = np.ones((64, 192, 192), dtype=bool)
        >>> tokens, coords, filtered = tokenize_volume(img_arr, mask_arr)
        >>> print(tokens.shape)  # (16*12*12, 4*16*16) = (2304, 1024)
        >>> print(coords.shape)  # (2304, 3)
        >>> print(filtered.shape)  # (2304,) - 0 for foreground, 1 for background
    
    Notes:
        - Image dimensions must be divisible by patch_size
        - Patch features = product of patch_size elements (e.g., 4*16*16=1024)
        - Coordinates represent the (d, h, w) index of each patch in the grid
        - A token is marked as background if ANY pixel in its patch is background
    """
    import torch
    from einops import rearrange
    
    D, H, W = img_arr.shape
    p1, p2, p3 = patch_size
    
    # Validate dimensions are divisible by patch size
    assert D % p1 == 0, f"Depth {D} not divisible by patch size {p1}"
    assert H % p2 == 0, f"Height {H} not divisible by patch size {p2}"
    assert W % p3 == 0, f"Width {W} not divisible by patch size {p3}"
    
    # Calculate number of patches per dimension
    n_patches_d = D // p1
    n_patches_h = H // p2
    n_patches_w = W // p3
    
    # Tokenize image: [D, H, W] -> [N, patch_features]
    tokens_torch = rearrange(
        torch.from_numpy(img_arr).unsqueeze(0),  # Add channel dim [1, D, H, W]
        "c (d p1) (h p2) (w p3) -> (d h w) (c p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    ).squeeze(1)  # Remove channel dim: [N, 1024]
    
    # Tokenize mask: a patch is background if ANY pixel in it is background
    # mask_arr is True for foreground, False for background
    mask_tokens = rearrange(
        torch.from_numpy(mask_arr),
        "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    )
    # filtered: 1 if any pixel in patch is background (i.e., not all pixels are foreground)
    # 0 if all pixels in patch are foreground
    filtered = (~(mask_tokens.all(dim=1))).to(torch.uint8)  # [N]
    
    # Generate 3D coordinates for each token
    coords_d, coords_h, coords_w = np.meshgrid(
        np.arange(n_patches_d),
        np.arange(n_patches_h),
        np.arange(n_patches_w),
        indexing='ij'
    )
    coords_torch = torch.from_numpy(
        np.stack([coords_d.flatten(), coords_h.flatten(), coords_w.flatten()], axis=1)
    ).long()
    
    if remove_background:
        # Physically remove background tokens
        fg_mask = ~filtered.bool()
        tokens = tokens_torch[fg_mask].numpy()
        coords = coords_torch[fg_mask].numpy()
        filtered_out = np.array([], dtype=np.uint8)  # Empty since we removed them
    else:
        # Keep all tokens, return filtered mask
        tokens = tokens_torch.numpy()
        coords = coords_torch.numpy()
        filtered_out = filtered.numpy()
    
    return tokens, coords, filtered_out