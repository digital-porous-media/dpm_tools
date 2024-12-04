import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pytest
import porespy as ps
import dpm_tools.io as dpm_io
import dpm_tools.metrics as dpm_metrics
import dpm_tools.visualization as dpm_vis
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")


class TestVisualization:
    def setup_class(self):
        np.random.seed(12573)
        spheres = ps.generators.overlapping_spheres(
            shape=[100, 100, 100], r=10, porosity=0.6).astype(np.uint8)
        self.spheres = dpm_io.Image(scalar=spheres)
        self.thickness = dpm_io.Image(
            scalar=dpm_metrics.edt(self.spheres.scalar))
        self.velocity = dpm_io.Image(scalar=spheres, vector=[
                                     self.thickness.scalar] * 3)

    def test_hist(self):
        plt.close('all')
        n_figs_before = plt.gcf().number
        single_hist_fig = dpm_vis.hist(self.spheres, nbins=5)
        n_figs_after = single_hist_fig.number
        assert n_figs_before < n_figs_after

        patch_height = [patch.get_height()
                        for patch in single_hist_fig.gca().patches]
        assert_allclose(
            patch_height, [2.0019, 0.0, 0.0, 0.0, 2.9981000000000004])

    def test_plot_slice(self):
        n_figs_before = plt.gcf().number
        slice_fig = dpm_vis.plot_slice(self.thickness)
        n_figs_after = slice_fig.number
        assert n_figs_before < n_figs_after

    def test_make_thumbnail(self):
        n_figs_before = plt.gcf().number
        thumbnail_fig = dpm_vis.make_thumbnail(self.thickness)
        n_figs_after = thumbnail_fig.number
        assert n_figs_before < n_figs_after

    def test_make_gif(self):
        gif_images = dpm_vis.make_gif(self.thickness)
        assert len(gif_images) > 0

    def test_plot_heterogeneity_curve(self):
        radii, variances = dpm_metrics.heterogeneity_curve(self.spheres.scalar)
        n_figs_before = plt.gcf().number
        dpm_vis.plot_heterogeneity_curve(radii, variances)
        n_figs_after = plt.gcf().number
        assert n_figs_before < n_figs_after

    def test_orthogonal_slices(self):
        fig = dpm_vis.orthogonal_slices(self.spheres)
        assert len(fig.mesh) == 3
        assert isinstance(fig.mesh[0], pv.PolyData)
        assert isinstance(fig.mesh[1], pv.PolyData)
        assert isinstance(fig.mesh[2], pv.PolyData)

        assert fig.mesh[0].n_cells
        assert fig.mesh[1].n_cells
        assert fig.mesh[2].n_cells

    def test_plot_isosurface(self):
        fig = dpm_vis.plot_isosurface(self.spheres, show_isosurface=[0.5])
        assert isinstance(fig.mesh, pv.PolyData)

        assert fig.mesh.n_cells

    def test_bounding_box(self):
        fig = dpm_vis.bounding_box(self.spheres)
        assert isinstance(fig.mesh, pv.PolyData)
        assert fig.mesh.n_cells

    def test_plot_glyph(self):
        fig = dpm_vis.plot_glyph(self.velocity)
        assert isinstance(fig.mesh, pv.PolyData)
        assert fig.mesh.n_cells

    def test_plot_streamlines(self):
        fig = dpm_vis.plot_streamlines(self.velocity)
        assert isinstance(fig.mesh, pv.PolyData)
        assert fig.mesh.n_cells

    def test_plot_scalar_volume(self):
        fig = dpm_vis.plot_scalar_volume(self.thickness)
        assert isinstance(fig.volume, pv.Volume)

    def test_plot_medial_axis(self):
        fig = dpm_vis.plot_medial_axis(self.spheres)
        assert isinstance(fig.mesh, pv.PolyData)
        assert fig.mesh.n_cells


if __name__ == "__main__":
    tests = TestVisualization()
    tests.setup_class()
    for item in tests.__dir__():
        if item.startswith('test'):
            print('running test: ' + item)
            tests.__getattribute__(item)()
