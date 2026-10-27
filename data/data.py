import torch
import numpy as np
import SimpleITK as sitk


# 图像读取
def img_read(
        img_path,
        target_spacing=(1, 1, 1), # 采样间距 (z, y, x), mm
        orient="LPS",
        interpolate_mode = "trilinear"  # trilinear | nearest
):
    # 图像读取
    img = sitk.ReadImage(str(img_path))
    img_size = img.GetSize() # (x, y, z)
    # 坐标系处理
    native_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection()
    )
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orient)
    img = orient_filter.Execute(img)
    # 尺度处理
    native_spacing = list(img.GetSpacing())
    arr = sitk.GetArrayFromImage(img)
    # arr shape is (D, H, W), native_spacing is (x, y, z)
    z, y, x = native_spacing[2], native_spacing[1], native_spacing[0]
    target_size = (
        int(arr.shape[0] * z / target_spacing[0]),  # D
        int(arr.shape[1] * y / target_spacing[1]),  # H
        int(arr.shape[2] * x / target_spacing[2]),  # W
    )
    # Resample via interpolation
    arr_t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
    arr_t = torch.nn.functional.interpolate(arr_t, size=target_size, mode=interpolate_mode).squeeze()
    
    return arr_t.numpy()