import numpy as np
try:
    import dpm_srg
    import dpm_srm
    
except ImportError as e:
    raise ImportError("Segmentation module requires additional dependencies. "
                      "Install them using 'pip install dpm-tools[segment]'") from e


    
    

def statistical_region_merging(image: np.ndarray, Q: float = 5.0, normalize: bool = True):
    """ 
    Perform statistical region merging on a gray level image
    
    Parameters:
        image: A 2D or 3D numpy.ndarray representing the gray level image.
        Q: Parameter controlling the intensity difference threshold for merging adjacent regions. 
        Roughly speaking, Q is an estimate of the number of expected regions, though this is not strictly adhered to. 
        The larger the Q value, the more regions are produced.
        normalize: If True, normalize the input image to the range [0, max(dtype)]. Default is True

    Returns:
        numpy.ndarray: A numpy.ndarray labeled image of the same shape and datatype as the input image.
    """
    # Get image dimensions
    image_dims = str(image.ndims)
    assert image_dims in ["2", "3"], "Image must be either 2 or 3 dimensional ndarray"
    
    # Get image datatype
    image_dtype = str(image.dtype)
    assert image_dtype in ["uint8", "uint16", "uint32"], "Unsupported image datatype, expected uint8, uint16 or uint32"
    
    # Normalize image if necessary
    if normalize:
        img_min = np.percentile(image, 0.01)
        img_max = np.percentile(image, 99.99)
        image[image < img_min] = img_min
        image[image > img_max] = img_max
        image = (image - img_min) / (img_max - img_min) * np.iinfo(image.dtype).max
    
    func_dict = {"2":
                    {
                        "uint8": dpm_srm.SRM2D_u8,
                        "uint16": dpm_srm.SRM2D_u16,
                        "uint32": dpm_srm.SRM2D_u32,
                    },
                 "3":
                    {
                        "uint8": dpm_srm.SRM3D_u8,
                        "uint16": dpm_srm.SRM3D_u16,
                        "uint32": dpm_srm.SRM3D_u32,
                    }
                }
    
    srg_obj = func_dict[image_dims][image_dtype](image, Q)
    srg_obj.segment()
    
    return srg_obj.get_result()
    

def seeded_region_growing(image: np.ndarray, seed_image: np.ndarray, normalize: bool = True)
    """ Perform seeded region growing on a gray level image using predefined seeds
    """
    # TODO: Write wrapper code for dpm_srg
    image_dims = image.ndims
    assert image_dims in [2, 3], "Image must be either 2 or 3 dimensional ndarray"
    
    seed_dims = seed_image.ndims
    assert seed_dims == image_dims, "Seed image must have the same dimensions as the input image"
    
    # Get image datatype
    image_dtype = str(image.dtype)
    assert image_dtype in ["uint8", "uint16", "uint32"], "Unsupported image datatype, expected uint8, uint16 or uint32"
    assert seed_image.dtype == np.uint8, "Seed image must have uint8 datatype"
    
    # Normalize image if necessary
    if normalize:
        img_min = np.percentile(image, 0.01)
        img_max = np.percentile(image, 99.99)
        image[image < img_min] = img_min
        image[image > img_max] = img_max
        image = (image - img_min) / (img_max - img_min) * np.iinfo(image.dtype).max
    
    func_dict = {"2":
                    {
                        "uint8": dpm_srg.SRG2D_u8,
                        "uint16": dpm_srg.SRG2D_u16,
                        "uint32": dpm_srg.SRG2D_u32,
                    },
                 "3":
                    {
                        "uint8": dpm_srg.SRG3D_u8,
                        "uint16": dpm_srg.SRG3D_u16,
                        "uint32": dpm_srg.SRG3D_u32,
                    }
                }
    
    srg_obj = func_dict[image_dims][image_dtype](image, Q)
    srg_obj.segment()
    
    return srg_obj.get_result()
