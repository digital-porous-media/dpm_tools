import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib.pyplot as plt
from dpm_tools.segmentation import seeded_region_growing as srg, statistical_region_merging as srm
# import pyvista as pv
import warnings
warnings.filterwarnings("ignore")

class TestSegmentation:
    def setup_class(self):
        np.random.seed(130621)
        image = np.random.randint(0, 256, size=(10, 10, 10), dtype=np.uint8)
        seeds = np.random.randint(0, 5, size=(10, 10, 10), dtype=np.uint8)
        self.image3d = image
        self.seeds3d = seeds
        self.image2d = image[0]
        self.seeds2d = seeds[0]

    def test_srg2d_u8(self):
        seg_result = srg(self.image2d, self.seeds2d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [4, 2, 4, 3, 4, 2, 4, 1, 4, 4],
                            [4, 2, 2, 4, 3, 4, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 2],
                            [4, 3, 3, 3, 3, 3, 4, 3, 2, 1],
                            [1, 4, 4, 3, 3, 3, 3, 1, 2, 3],
                            [2, 1, 4, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 4, 4, 3, 2, 3],
                            [4, 4, 2, 2, 4, 2, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        assert_allclose(seg_result, seg_true)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds2d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds2d)) and np.all(seg_result <= np.amax(self.seeds2d)), "Segment labels should be between 0 and 4"
    
    def test_srg2d_u16(self):
        seg_result = srg(self.image2d.astype(np.uint16), self.seeds2d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [4, 2, 4, 3, 4, 2, 4, 1, 4, 4],
                            [4, 2, 2, 4, 3, 4, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 2],
                            [4, 3, 3, 3, 3, 3, 4, 3, 2, 1],
                            [1, 4, 4, 3, 3, 3, 3, 1, 2, 3],
                            [2, 1, 4, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 4, 4, 3, 2, 3],
                            [4, 4, 2, 2, 4, 2, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        assert_allclose(seg_result, seg_true)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds2d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds2d)) and np.all(seg_result <= np.amax(self.seeds2d)), "Segment labels should be between 0 and 4"
    
    def test_srg2d_u32(self):
        seg_result = srg(self.image2d.astype(np.uint32), self.seeds2d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [4, 2, 4, 3, 4, 2, 4, 1, 4, 4],
                            [4, 2, 2, 4, 3, 4, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 2],
                            [4, 3, 3, 3, 3, 3, 4, 3, 2, 1],
                            [1, 4, 4, 3, 3, 3, 3, 1, 2, 3],
                            [2, 1, 4, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 4, 4, 3, 2, 3],
                            [4, 4, 2, 2, 4, 2, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        assert_allclose(seg_result, seg_true)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds2d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds2d)) and np.all(seg_result <= np.amax(self.seeds2d)), "Segment labels should be between 0 and 4"
    
    def test_srg3d_u8(self):
        seg_result = srg(self.image3d, self.seeds3d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [2, 2, 4, 3, 4, 2, 4, 4, 4, 4],
                            [4, 2, 2, 4, 4, 2, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 3],
                            [4, 3, 3, 3, 3, 3, 4, 3, 3, 1],
                            [1, 4, 4, 4, 3, 3, 3, 1, 2, 3],
                            [2, 2, 3, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 2, 4, 4, 2, 3],
                            [4, 4, 2, 2, 4, 4, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds3d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds3d)) and np.all(seg_result <= np.amax(self.seeds3d)), "Segment labels should be between 0 and 4"
        
    def test_srg3d_u16(self):
        seg_result = srg(self.image3d.astype(np.uint16), self.seeds3d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [2, 2, 4, 3, 4, 2, 4, 4, 4, 4],
                            [4, 2, 2, 4, 4, 2, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 3],
                            [4, 3, 3, 3, 3, 3, 4, 3, 3, 1],
                            [1, 4, 4, 4, 3, 3, 3, 1, 2, 3],
                            [2, 2, 3, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 2, 4, 4, 2, 3],
                            [4, 4, 2, 2, 4, 4, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds3d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds3d)) and np.all(seg_result <= np.amax(self.seeds3d)), "Segment labels should be between 0 and 4"
    
    def test_srg3d_u32(self):
        seg_result = srg(self.image3d.astype(np.uint32), self.seeds3d)
        seg_true = np.array([[4, 2, 1, 1, 2, 3, 1, 2, 2, 2],
                            [2, 2, 4, 3, 4, 2, 4, 4, 4, 4],
                            [4, 2, 2, 4, 4, 2, 2, 4, 2, 4],
                            [4, 1, 2, 2, 1, 2, 1, 2, 3, 3],
                            [4, 3, 3, 3, 3, 3, 4, 3, 3, 1],
                            [1, 4, 4, 4, 3, 3, 3, 1, 2, 3],
                            [2, 2, 3, 1, 1, 4, 3, 1, 3, 1],
                            [3, 3, 3, 3, 2, 2, 4, 4, 2, 3],
                            [4, 4, 2, 2, 4, 4, 3, 2, 3, 1],
                            [2, 4, 3, 3, 2, 4, 1, 1, 4, 2]])
        
        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        assert np.unique(seg_result).size <= np.unique(self.seeds3d).size, "Number of segments should not exceed number of seeds"
        assert np.all(seg_result >= np.amin(self.seeds3d)) and np.all(seg_result <= np.amax(self.seeds3d)), "Segment labels should be between 0 and 4"
    
    def test_srm2d_u8(self):
        seg_result = srm(self.image2d, Q=5.0)
        seg_true = np.array([[249,  78,  78,  78, 208, 208, 208,  78,  78,  78],
                            [249,  78,  78,  78, 208, 208, 208,  78,  78,  78],
                            [ 78,  78, 194, 194,  78,  78,  78, 229, 229,  78],
                            [ 78,  78,  78,  78, 195,  78,  78,  78,  78,  78],
                            [ 78,  78,  78, 195, 195,  78,  78,  78,  78,  78],
                            [ 78,  78, 210,  78, 195,  78,  78, 158, 158, 158],
                            [ 78,  78, 210,  78,  78,  78,  78,  78, 158, 158],
                            [234,  78, 210,  78, 238, 238,  78,  78, 158, 158],
                            [234,  78,  78,  78,  78, 238, 158, 158, 158, 158],
                            [234,  78,  78,  78,  78,  78, 158, 158, 158, 158]])
        # seg_true[seg_true == 78] = 79
                                    
        assert_allclose(seg_result, seg_true, atol=1.5)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
        
        
    def test_srm2d_u16(self):
        seg_result = srm(self.image2d.astype(np.uint16), Q=5.0)
        seg_true = np.array([[64498, 20863, 20863, 20863, 54049, 54049, 54049, 20863, 65531, 20863],
                            [64498, 20863, 20863, 20863, 54049, 54049, 54049, 20863, 20863, 20863],
                            [20863, 20863, 50250, 50250, 20863, 20863, 20863, 59446, 59446, 20863],
                            [20863, 20863, 20863, 20863, 50509, 20863, 20863, 20863, 20863, 20863],
                            [20863, 20863, 20863, 50509, 50509, 20863, 20863, 20863, 20863, 20863],
                            [20863, 20863, 54395, 20863, 50509, 20863, 20863, 41028, 41028, 41028],
                            [20863, 20863, 54395, 20863, 20863, 20863, 20863, 20863, 41028, 41028],
                            [60526, 20863, 54395, 20863, 61821, 61821, 20863, 20863, 41028, 41028],
                            [60526, 20863, 20863, 20863, 20863, 61821, 41028, 41028, 41028, 41028],
                            [60526, 20863, 20863, 20863, 20863, 20863, 41028, 41028, 41028, 41028]])
                                    
        assert_allclose(seg_result, seg_true, atol=1.5)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint16, "Segmented image should be of type uint16"
    @pytest.mark.skip(reason="Passes locally but runs out of memory on GitHub Actions")    
    def test_srm2d_u32(self):
        seg_result = srm(self.image2d.astype(np.uint32), Q=5.0)
        seg_true = np.array([[4159147423, 2359534120, 1120177789, 1323907596, 4023327551, 2325579152,  3955417615,  865515529, 4294631142,  525965849],
                            [4294967295,  678763205, 1340885080,  271303589, 3921462647, 4159147423,  2868858639, 1561592372, 2478376508, 1255997660],
                            [ 933425465,   84551265, 2716061284, 3870530195,   84551265, 1239020177,   763650625, 3904485163, 3887507679,  508988365],
                            [3191430835, 1748344696, 1425772500, 1289952628, 3955417615, 1765322180,  1188087725, 1289952628,  152461201, 2800948703],
                            [1188087725, 1748344696, 1901142052, 3785642775, 3055610963, 2070916892,  2393489088,  542943333,  967380433,  729695657],
                            [2970723543, 2376511604, 3191430835, 1001335401, 2444421540, 1357862564,  1069245337, 3700755355, 3089565931, 2478376508],
                            [ 373168493, 2172781796, 3768665291, 1188087725, 1239020177, 2070916892,  2325579152, 1289952628, 2834903671, 2631173864],
                            [4023327551, 1510659920, 3734710323,  118506233, 4159147423, 4210079875,    67573781,  695740689, 1493682436, 2970723543],
                            [4294967295, 2274646700,  678763205,   33618813, 2104871860, 3785642775,  1918119536, 3463070579, 2138826828, 2902813607],
                            [3581912967, 1782299664, 1765322180, 2410466572, 2359534120, 1646479792,  3225385803, 3293295739, 2682106316, 1510659920]])
                                    
        assert_allclose(seg_result, seg_true, atol=1.5)
        assert seg_result.shape == self.image2d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint32, "Segmented image should be of type uint32"
    def test_srm3d_u8(self):
        seg_result = srm(self.image3d, Q=5.0)
        seg_true = np.array([[179,  65,  65,  65, 179,  85, 179,  65,  65,  65],
                            [179,  65,  65,  65, 179, 179, 179,  65,  65,  65],
                            [ 65,  65, 179, 179,  65,  65,  65, 179, 179,  65],
                            [179, 179, 179, 179, 179,  65,  65,  65,  65,  65],
                            [179, 179, 179, 179, 179,  65,  65,  65,  65,  65],
                            [179, 179, 179,  65, 179,  65,  65, 179, 179, 179],
                            [ 21, 179, 179,  65,  65,  65,  65,  65, 179, 179],
                            [179, 179, 179,  65, 179, 179,  65,  65, 179, 179],
                            [179, 179,  65,  65, 179, 179,  65, 179, 179, 179],
                            [179, 179, 179, 179, 179, 179, 179, 179, 179, 179]])
        
        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint8, "Segmented image should be of type uint8"
    
    def test_srm3d_u16(self):
        seg_result = srm(self.image3d.astype(np.uint16), Q=5.0)
        seg_true = np.array([[46067, 17176, 17176, 17176, 46067, 46067, 46067, 17176, 17176, 17176],
                             [46067, 17176, 17176, 17176, 46067, 46067, 46067, 17176, 17176, 17176],
                             [17176, 17176, 46067, 46067, 17176, 17176, 17176, 46067, 46067, 17176],
                            [46067, 46067, 46067, 46067, 46067, 17176, 17176, 17176, 17176, 17176],
                            [46067, 46067, 46067, 46067, 46067, 17176, 17176, 17176, 17176, 17176],
                            [46067, 46067, 46067, 17176, 46067, 17176, 17176, 46067, 46067, 46067],
                            [ 5397, 46067, 46067, 17176, 17176, 17176, 17176, 17176, 46067, 46067],
                            [46067, 46067, 46067, 17176, 46067, 46067, 17176, 17176, 46067, 46067],
                            [46067, 46067, 17176, 17176, 46067, 46067, 17176, 46067, 46067, 46067],
                            [46067, 46067, 46067, 46067, 46067, 46067, 46067, 46067, 46067, 46067]], dtype=np.uint16)
        
        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint16, "Segmented image should be of type uint16"
    
    @pytest.mark.skip(reason="Passes locally but runs out of memory on GitHub Actions")    
    def test_srm3d_u32(self):
        seg_result = srm(self.image3d.astype(np.uint32), Q=5.0)
        
        seg_true = np.array([[4126537205, 2341178251, 1111638594, 1313754702, 3991793133, 2307492233, 3924421097,  858993459,          0,  522133279],
                            [4261281277,  673720360, 1330597711,  269488144, 3890735079, 4126537205, 2846468521, 1549556828, 2459079314, 1246382666],
                            [ 926365495,   84215045, 2694881440, 3840206052,   84215045, 1229539657,  757935405, 3873892070, 3857049061,  505290270],
                            [3166485692, 1734829927, 1414812756, 1280068684, 3924421097, 1751672936, 1179010630, 1280068684,  151587081, 2779096485],
                            [1179010630, 1734829927, 1886417008, 3755991007, 3031741620, 2054847098, 2374864269,  538976288,  960051513,  724249387],
                            [2947526575, 2358021260, 3166485692,  993737531, 2425393296, 1347440720, 1061109567, 3671775962, 3065427638, 2459079314],
                            [ 370546198, 2155905152, 3739147998, 1179010630, 1229539657, 2054847098, 2307492233, 1280068684, 2812782503, 2610666395],
                            [3991793133, 1499027801, 3705461980,  117901063, 4126537205, 4177066232,   67372036,  690563369, 1482184792, 2947526575],
                            [4261281277, 2256963206,  673720360,   33686018, 2088533116, 3755991007, 1903260017, 3435973836, 2122219134, 2880154539],
                            [3553874899, 1768515945, 1751672936, 2391707278, 2341178251, 1633771873, 3200171710, 3267543746, 2661195422, 1499027801]])
        

        assert_allclose(seg_result[0], seg_true)
        assert seg_result.shape == self.image3d.shape, "Segmented image should have the same shape as input"
        assert seg_result.dtype == np.uint32, "Segmented image should be of type uint32"


if __name__ == "__main__":
    tests = TestSegmentation()
    tests.setup_class()
    for item in tests.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            tests.__getattribute__(item)()