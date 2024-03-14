import numpy as np
import pytest
import dpm_tools
from pathlib import Path
from numpy.testing import assert_allclose
from skimage.morphology import ball, cube
import porespy as ps


class MetricsTest:
    def setup_class(self):
        self.ball = ball(10)
        np.random.seed(12573)
        self.spheres = ps.generators.overlapping_spheres(shape=[100, 100, 100], r=10, porosity=0.6)

    def test_edt_ball(self):
        euclidean_distance = dpm_tools.metrics.edt(self.ball)
        assert_allclose([euclidean_distance.min(), euclidean_distance.max()], [0.0, 10.049875])

    def test_sdt_ball(self):
        signed_distance = dpm_tools.metrics.sdt(self.ball)
        assert_allclose([signed_distance.min(), signed_distance.max()], [-7.5498343, 10.049875])

    def test_mis_ball(self):
        inscribed_sphere = dpm_tools.metrics.mis(self.ball)
        assert_allclose([inscribed_sphere.min(), inscribed_sphere.max()], [0.0, 10.049875])

    def test_slicewise_edt_ball(self):
        slicewise_euclidean_distance = dpm_tools.metrics.slicewise_edt(self.ball)
        max_vals = np.array([ 1.,  4.472136, 6.0827627, 7.2111025, 8.062258,
                            8.944272, 9.219544, 9.848858, 9.848858, 10.,
                            10.049875, 10.,  9.848858,  9.848858,  9.219544,
                            8.944272,  8.062258,  7.2111025,  6.0827627,  4.472136, 1.])

        assert_allclose(np.max(slicewise_euclidean_distance, axis=(0,1)), max_vals)

    def test_slicewise_mis_ball(self):
        slicewise_inscribed_spheres = dpm_tools.metrics.slicewise_mis(self.ball)
        max_vals = np.array([1, 4, 6, 7, 7, 8, 9, 9, 9, 10, 10, 10, 9, 9, 9, 8, 7, 7, 6, 4, 1, 1])

        assert_allclose(np.max(slicewise_inscribed_spheres, axis=(0, 1)), max_vals)

    def test_chords_ball(self):
        x, y, areas = dpm_tools.metrics.chords(self.ball)
        chords_max = np.array([ 1.,  9., 13., 15., 17., 17., 19., 19., 19., 19., 21., 19., 19.,
                    19., 19., 17., 17., 15., 13.,  9.,  1.])

        assert_allclose(np.max(x, axis=(0, 1)), chords_max)
        assert_allclose(np.max(y, axis=(1, 2)), chords_max)
        assert_allclose(np.amax(areas), np.pi * (np.amax(chords_max/2)**2))

    def test_time_of_flight_ball(self):
        tofl_trended = dpm_tools.metrics.time_of_flight(self.ball, boundary='l', detrend=False)
        tofr_trended = dpm_tools.metrics.time_of_flight(self.ball, boundary='r', detrend=False)

        assert_allclose([tofl_trended.max(), tofr_trended.max()], [19.5, 19.5])

        tofl_detrended = dpm_tools.metrics.time_of_flight(self.ball, boundary='l', detrend=True)
        tofr_detrended = dpm_tools.metrics.time_of_flight(self.ball, boundary='r', detrend=True)

        assert_allclose([tofl_detrended.min(), tofl_detrended.max(), tofr_detrended.min(), tofr_detrended.max()],
                           [0.0, 6.18626252724756, 0.0, 6.18626252724756])
    def test_constriction_factor_ball(self):
        _, _, thickness_map = dpm_tools.metrics.chords(self.ball)
        cf = dpm_tools.metrics.constriction_factor(thickness_map, power=1)

        assert_allclose(np.unique(cf), np.array([0., 0.11111112, 0.14285713, 0.33333334, 0.42857137,
                                                    0.4285714, 0.63636357, 0.6363636, 0.6923076, 0.6923077,
                                                    0.73333335, 0.7333334, 0.7777778, 0.77777785, 0.8181817,
                                                    0.81818175, 0.8181818, 0.8461538, 0.84615386, 0.8461539,
                                                    0.8666667, 0.86666673, 0.8823528 , 0.8823529, 0.88235295,
                                                    0.8947368, 0.8947369, 0.90476185, 0.9047619, 1.,
                                                    1.1052631, 1.1052632, 1.1176469, 1.117647, 1.1333333,
                                                    1.1333334, 1.153846, 1.1538461, 1.1818181, 1.1818182,
                                                    1.2222222, 1.2222223, 1.2857141, 1.2857143, 1.3636363,
                                                    1.3636364, 1.4444444, 1.4444445, 1.5714285, 1.5714287,
                                                    2.3333335, 3., 7., 8.999999]))

        thickness_map = dpm_tools.metrics.slicewise_edt(self.ball)
        cf = dpm_tools.metrics.constriction_factor(thickness_map, power=2)

        edt_unique = np.array([ 0.        ,  0.05      ,  0.12500001,  0.2       ,  0.22222221,
                                0.25      ,  0.30769232,  0.3125    ,  0.3846154 ,  0.39999998,
                                0.4       ,  0.44444445,  0.4705882 ,  0.49999997,  0.5       ,
                                0.50000006,  0.5294118 ,  0.54054046,  0.5555556 ,  0.5625    ,
                                0.5862069 ,  0.5882353 ,  0.6153846 ,  0.625     ,  0.62500006,
                                0.64      ,  0.65      ,  0.68      ,  0.6896552 ,  0.6923077 ,
                                0.71153855,  0.7199999 ,  0.7222222 ,  0.7222223 ,  0.725     ,
                                0.7352942 ,  0.7551021 ,  0.75555545,  0.7647059 ,  0.76470596,
                                0.7692308 ,  0.7758621 ,  0.78125   ,  0.78125006,  0.7837837 ,
                                0.79999995,  0.8       ,  0.8055556 ,  0.8125    ,  0.81632656,
                                0.82      ,  0.8222223 ,  0.828125  ,  0.8472224 ,  0.8499999 ,
                                0.85      ,  0.86206895,  0.87628865,  0.88888884,  0.8965517 ,
                                0.8999999 ,  0.9       ,  0.90625006,  0.9111111 ,  0.9183673 ,
                                0.9189187 ,  0.9245283 ,  0.925     ,  0.9411765 ,  0.94117653,
                                0.94444436,  0.94444454,  0.9508196 ,  0.95312506,  0.96153843,
                                0.97      ,  0.97297287,  0.9756098 ,  0.98      ,  0.9846154 ,
                                0.9876543 ,  0.9878048 ,  0.9900991 ,  1.        ,  1.0099999 ,
                                1.0123458 ,  1.0125    ,  1.015625  ,  1.0204082 ,  1.025     ,
                                1.0277779 ,  1.0309278 ,  1.04      ,  1.0491803 ,  1.0517242 ,
                                1.0588235 ,  1.0588237 ,  1.0625    ,  1.081081  ,  1.0816326 ,
                                1.0882355 ,  1.0888889 ,  1.097561  ,  1.1034482 ,  1.1111112 ,
                                1.1111113 ,  1.1153846 ,  1.1250001 ,  1.1411765 ,  1.16      ,
                                1.1764706 ,  1.1764708 ,  1.1803277 ,  1.2075472 ,  1.2162161 ,
                                1.2195122 ,  1.225     ,  1.2307693 ,  1.2413793 ,  1.25      ,
                                1.2500001 ,  1.2758622 ,  1.28      ,  1.2888889 ,  1.3       ,
                                1.3076922 ,  1.3076923 ,  1.3235296 ,  1.3243241 ,  1.3599999 ,
                                1.3793104 ,  1.3846152 ,  1.3846154 ,  1.3888891 ,  1.4054053 ,
                                1.4444444 ,  1.45      ,  1.4705882 ,  1.5384616 ,  1.5625    ,
                                1.5999999 ,  1.6       ,  1.6250001 ,  1.7       ,  1.7058823 ,
                                1.7777778 ,  1.8       ,  1.8500001 ,  1.8888888 ,  1.9999999 ,
                                2.        ,  2.0000002 ,  2.1250002 ,  2.25      ,  2.5       ,
                                2.5000002 ,  2.6       ,  3.2       ,  3.25      ,  4.        ,
                                4.5000005 ,  5.        ,  7.9999995 , 20.        ])

        assert_allclose(np.unique(cf), edt_unique)

    def test_minkowski_2d_ball(self):
        with pytest.raises(Exception):
            dpm_tools.metrics.minkowski_2d(self.ball)

        area, perim, curv = dpm_tools.metrics.minkowski_2d(self.ball[10])
        assert_allclose([area, perim, curv], (317.0, 65.06451422842865, 1.0))

    def test_minkowski_3d_ball(self):
        volume, area, curv, euler = dpm_tools.metrics.minkowski_3d(self.ball)
        assert_allclose([volume, area, curv, euler],
                           [4169.0, 1262.1090288786693, 129.16729594427238, 1.9999999999999998])

    def test_morph_drain(self):


        radii, sw, config = dpm_tools.metrics.morph_drain(self.spheres)

        assert_allclose(radii, np.array([17.36188383, 16.49378964, 15.66910016, 14.88564515, 14.14136289,
                                            13.43429475, 12.76258001, 12.12445101, 11.51822846, 10.94231704,
                                            10.39520118,  9.87544113,  9.38166907,  8.91258562,  8.46695633,
                                             8.04360852,  7.64142809,  7.25935669,  6.89638885,  6.55156941,
                                             6.22399094,  5.91279139]))

        assert_allclose(sw, np.array([1.        , 0.99165638, 0.98242387, 0.96848671, 0.96848671,
                                         0.96361529, 0.95676428, 0.95676428, 0.9234682 , 0.72262266,
                                         0.72262266, 0.53219706, 0.53219706, 0.38310597, 0.38310597,
                                         0.38310597, 0.19660618, 0.19660618, 0.1139972 , 0.1139972 ,
                                         0.1139972 , 0.06980421]))

        ids, counts = np.unique(config, return_counts=True)
        assert_allclose(ids, [0, 1, 2])
        assert_allclose(counts, [400380, 557764,  41856])

    def test_morph_drain_config_ball(self):
        config, sw = dpm_tools.metrics._morph_drain_config(self.spheres, 4.5)
        assert_allclose(sw, 0.04025883059270872)

        ids, counts = np.unique(config, return_counts=True)
        assert_allclose(ids, [0, 1, 2])
        assert_allclose(counts, [400380, 575480,  24140])

    def test_heterogeneity_curve_ball(self):
        np.random.seed(112057)
        radii, variance = dpm_tools.metrics.heterogeneity_curve(self.spheres)
        assert_allclose(radii, np.array([ 18,  20,  22,  24,  26,  28,  30,  32,  34,  36,  38,  40,  42,
                                             44,  46,  48,  50,  52,  55,  57,  59,  61,  63,  65,  67,  69,
                                             71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,
                                             97,  99, 101, 103, 106, 108, 110, 112, 114, 116, 118]))

        assert_allclose(variance, np.array([1.51746246e-02, 8.78398359e-03, 6.71367477e-03, 4.71184115e-03,
                                               3.78188518e-03, 2.20582610e-03, 1.55966288e-03, 1.04812750e-03,
                                               5.47927765e-04, 3.89760525e-04, 1.71487358e-04, 8.81955332e-05,
                                               7.27613964e-05, 3.85910879e-05, 1.85092052e-05, 1.53797914e-06,
                                               0.00000000e+00, 1.23259516e-32, 0.00000000e+00, 3.08148791e-33,
                                               3.08148791e-33, 0.00000000e+00, 3.08148791e-33, 0.00000000e+00,
                                               7.70371978e-34, 7.70371978e-34, 3.08148791e-33, 7.70371978e-34,
                                               7.70371978e-34, 7.70371978e-34, 7.70371978e-34, 0.00000000e+00,
                                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.92592994e-34,
                                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.81482486e-35,
                                               1.92592994e-34, 4.81482486e-35, 4.81482486e-35, 0.00000000e+00,
                                               0.00000000e+00, 4.81482486e-35]))

if __name__ == "__main__":
    tests = MetricsTest()
    tests.setup_class()
    for item in tests.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            tests.__getattribute__(item)()