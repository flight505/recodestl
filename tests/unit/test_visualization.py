"""Unit tests for visualization functionality."""

from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from recodestl.visualization import (
    PointCloudVisualizer,
    plot_point_cloud,
    save_point_cloud_image,
    compare_sampling_methods,
)


@pytest.fixture
def sample_points():
    """Generate sample point cloud."""
    # Create a cube-like point cloud
    n_points = 100
    points = np.random.rand(n_points, 3) * 2 - 1
    return points


@pytest.fixture
def visualizer():
    """Create visualizer instance."""
    return PointCloudVisualizer(figsize=(8, 6))


class TestPointCloudVisualizer:
    """Test PointCloudVisualizer class."""
    
    def test_init(self):
        """Test visualizer initialization."""
        viz = PointCloudVisualizer(figsize=(10, 8))
        assert viz.figsize == (10, 8)
        
    def test_plot_points_3d(self, visualizer, sample_points):
        """Test 3D point plotting."""
        fig = visualizer.plot_points_3d(
            sample_points,
            title="Test Points",
            size=2.0,
            alpha=0.8,
        )
        
        assert fig is not None
        # There may be 2 axes if colorbar is included
        assert len(fig.axes) >= 1
        # First axis should be the 3D plot
        assert fig.axes[0].get_title() == "Test Points"
        plt.close(fig)
        
    def test_plot_points_3d_with_color(self, visualizer, sample_points):
        """Test 3D plotting with custom colors."""
        # Color by Z coordinate
        colors = sample_points[:, 2]
        
        fig = visualizer.plot_points_3d(
            sample_points,
            title="Colored Points",
            color=colors,
        )
        
        assert fig is not None
        plt.close(fig)
        
    def test_plot_with_normals(self, visualizer, sample_points):
        """Test plotting with normal vectors."""
        # Generate random normals
        normals = np.random.randn(len(sample_points), 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        fig = visualizer.plot_with_normals(
            sample_points,
            normals,
            title="Points with Normals",
            normal_length=0.1,
            subsample=10,
        )
        
        assert fig is not None
        plt.close(fig)
        
    def test_compare_point_clouds(self, visualizer, sample_points):
        """Test point cloud comparison."""
        # Create variations
        clouds = {
            "Original": sample_points,
            "Scaled": sample_points * 0.8,
            "Shifted": sample_points + 0.2,
        }
        
        fig = visualizer.compare_point_clouds(
            clouds,
            title="Comparison Test",
        )
        
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)
        
    def test_plot_sampling_density(self, visualizer, sample_points):
        """Test density visualization."""
        fig = visualizer.plot_sampling_density(
            sample_points,
            title="Density Test",
            bins=20,
        )
        
        assert fig is not None
        # Should have at least 4 axes (3 2D plots + 1 3D plot)
        assert len(fig.axes) >= 4
        plt.close(fig)
        
    def test_save_figure(self, visualizer, sample_points, tmp_path):
        """Test figure saving."""
        fig = visualizer.plot_points_3d(sample_points)
        output_path = tmp_path / "test_plot.png"
        
        visualizer.save_figure(fig, output_path, dpi=100)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)
        
    def test_axes_equal(self, visualizer):
        """Test equal axes setting."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Set some data
        ax.scatter([0, 1], [0, 2], [0, 3])
        
        # Apply equal axes
        visualizer._set_axes_equal(ax)
        
        # Check that ranges are equal
        x_range = np.diff(ax.get_xlim3d())[0]
        y_range = np.diff(ax.get_ylim3d())[0]
        z_range = np.diff(ax.get_zlim3d())[0]
        
        assert np.allclose(x_range, y_range)
        assert np.allclose(y_range, z_range)
        plt.close(fig)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_plot_point_cloud(self, sample_points, tmp_path):
        """Test quick plotting function."""
        # Test matplotlib plot
        fig = plot_point_cloud(
            sample_points,
            title="Quick Plot",
            interactive=False,
        )
        
        assert fig is not None
        plt.close(fig)
        
        # Test with save
        output_path = tmp_path / "quick_plot.png"
        fig = plot_point_cloud(
            sample_points,
            title="Quick Plot",
            save_path=output_path,
            interactive=False,
        )
        
        assert output_path.exists()
        plt.close(fig)
        
    def test_save_point_cloud_image(self, sample_points, tmp_path):
        """Test multi-view saving."""
        output_path = tmp_path / "multi_view.png"
        
        save_point_cloud_image(
            sample_points,
            output_path,
            title="Multi-View Test",
            views=[(30, 45), (90, 0)],
            dpi=100,
        )
        
        # Check that files were created
        expected_files = [
            tmp_path / "multi_view_elev30_azim45.png",
            tmp_path / "multi_view_elev90_azim0.png",
        ]
        
        for file in expected_files:
            assert file.exists()
            assert file.stat().st_size > 0
            
    def test_compare_sampling_methods(self, sample_points, tmp_path):
        """Test sampling comparison."""
        results = {
            "Method A": sample_points,
            "Method B": sample_points * 0.9,
            "Method C": sample_points + 0.1,
        }
        
        output_path = tmp_path / "comparison.png"
        
        fig = compare_sampling_methods(
            results,
            output_path=output_path,
            title="Method Comparison",
        )
        
        assert fig is not None
        assert output_path.exists()
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_points(self, visualizer):
        """Test with empty point cloud."""
        empty_points = np.array([]).reshape(0, 3)
        
        # Should not crash
        fig = visualizer.plot_points_3d(empty_points)
        assert fig is not None
        plt.close(fig)
        
    def test_single_point(self, visualizer):
        """Test with single point."""
        single_point = np.array([[0, 0, 0]])
        
        fig = visualizer.plot_points_3d(single_point)
        assert fig is not None
        plt.close(fig)
        
    def test_2d_points(self, visualizer):
        """Test with 2D points (should fail)."""
        points_2d = np.random.rand(10, 2)
        
        with pytest.raises(IndexError):
            visualizer.plot_points_3d(points_2d)
            
    def test_invalid_style(self):
        """Test with invalid matplotlib style."""
        # Should not crash with invalid style
        viz = PointCloudVisualizer(style="invalid_style_name")
        assert viz is not None