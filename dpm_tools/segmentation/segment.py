try:
    import dpm_srg
    import dpm_srm
    
except ImportError as e:
    raise ImportError("Segmentation module requires additional dependencies. "
                      "Install them using 'pip install dpm-tools[segment]'") from e
    
    

def statistical_region_merging(image, Q=5.0):
    """ Perform statistical region merging on a gray level image
    
    """
    # TODO: Write wrapper code for dpm_srm
    pass

def seeded_region_growing(image, seed_image):
    """ Perform seeded region growing on a gray level image using predefined seeds
    """
    # TODO: Write wrapper code for dpm_srg
    pass
