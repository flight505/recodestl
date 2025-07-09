"""Point cloud visualization tools."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PointCloudVisualizer:
    """Visualizer for point clouds and meshes."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = "seaborn-v0_8"):
        """Initialize visualizer.
        
        Args:
            figsize: Figure size for matplotlib
            style: Matplotlib style
        """
        self.figsize = figsize
        if style in plt.style.available:
            plt.style.use(style)
        
    def plot_points_3d(
        self,
        points: np.ndarray,
        title: str = "Point Cloud",
        color: Optional[np.ndarray] = None,
        size: float = 1.0,
        alpha: float = 0.8,
        elev: float = 30,
        azim: float = 45,
    ) -> Figure:
        """Plot 3D point cloud using matplotlib.
        
        Args:
            points: Point cloud array (N, 3)
            title: Plot title
            color: Color array (N,) or (N, 3) or single color
            size: Point size
            alpha: Point transparency
            elev: Elevation viewing angle
            azim: Azimuth viewing angle
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Handle color
        if color is None:
            # Color by height (Z coordinate)
            color = points[:, 2]
            
        # Plot points
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1], 
            points[:, 2],
            c=color,
            s=size,
            alpha=alpha,
            cmap='viridis',
        )
        
        # Add colorbar if using scalar colors
        if isinstance(color, np.ndarray) and color.ndim == 1:
            plt.colorbar(scatter, ax=ax, pad=0.1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Equal aspect ratio
        self._set_axes_equal(ax)
        
        plt.tight_layout()
        return fig
        
    def plot_points_interactive(
        self,
        points: np.ndarray,
        title: str = "Point Cloud",
        color: Optional[np.ndarray] = None,
        size: float = 2.0,
    ) -> Optional['go.Figure']:
        """Create interactive 3D plot using Plotly.
        
        Args:
            points: Point cloud array (N, 3)
            title: Plot title
            color: Color array or single color
            size: Point size
            
        Returns:
            Plotly figure or None if not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not installed. Use 'pip install plotly' for interactive plots.")
            return None
            
        # Handle color
        if color is None:
            color = points[:, 2]  # Color by height
            
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Z") if isinstance(color, np.ndarray) else None,
            ),
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
            ),
            width=800,
            height=600,
        )
        
        return fig
        
    def plot_with_normals(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        title: str = "Point Cloud with Normals",
        normal_length: float = 0.1,
        subsample: int = 10,
    ) -> Figure:
        """Plot point cloud with normal vectors.
        
        Args:
            points: Point cloud array (N, 3)
            normals: Normal vectors (N, 3)
            title: Plot title
            normal_length: Length of normal arrows
            subsample: Show every nth normal for clarity
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c='blue',
            s=1,
            alpha=0.5,
        )
        
        # Plot normals (subsample for clarity)
        indices = np.arange(0, len(points), subsample)
        for i in indices:
            ax.quiver(
                points[i, 0],
                points[i, 1],
                points[i, 2],
                normals[i, 0],
                normals[i, 1],
                normals[i, 2],
                length=normal_length,
                color='red',
                alpha=0.6,
                arrow_length_ratio=0.3,
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        self._set_axes_equal(ax)
        plt.tight_layout()
        return fig
        
    def compare_point_clouds(
        self,
        point_clouds: Dict[str, np.ndarray],
        title: str = "Point Cloud Comparison",
        subplot_shape: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Compare multiple point clouds side by side.
        
        Args:
            point_clouds: Dictionary mapping names to point arrays
            title: Overall figure title
            subplot_shape: (rows, cols) for subplot layout
            
        Returns:
            Matplotlib figure
        """
        n_clouds = len(point_clouds)
        
        # Determine subplot layout
        if subplot_shape is None:
            cols = min(3, n_clouds)
            rows = (n_clouds + cols - 1) // cols
        else:
            rows, cols = subplot_shape
            
        # Create figure
        fig = plt.figure(figsize=(self.figsize[0] * cols / 2, self.figsize[1] * rows / 2))
        fig.suptitle(title, fontsize=16)
        
        # Plot each point cloud
        for i, (name, points) in enumerate(point_clouds.items()):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Color by Z coordinate
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=points[:, 2],
                s=1,
                cmap='viridis',
            )
            
            ax.set_title(f"{name}\n({len(points)} points)")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            self._set_axes_equal(ax)
            
        plt.tight_layout()
        return fig
        
    def plot_sampling_density(
        self,
        points: np.ndarray,
        mesh_vertices: Optional[np.ndarray] = None,
        title: str = "Sampling Density",
        bins: int = 50,
    ) -> Figure:
        """Visualize sampling density as 2D histograms.
        
        Args:
            points: Sampled points
            mesh_vertices: Original mesh vertices for comparison
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # XY projection
        axes[0, 0].hist2d(points[:, 0], points[:, 1], bins=bins, cmap='hot')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('XY Density')
        axes[0, 0].set_aspect('equal')
        
        # XZ projection
        axes[0, 1].hist2d(points[:, 0], points[:, 2], bins=bins, cmap='hot')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        axes[0, 1].set_title('XZ Density')
        axes[0, 1].set_aspect('equal')
        
        # YZ projection
        axes[1, 0].hist2d(points[:, 1], points[:, 2], bins=bins, cmap='hot')
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].set_title('YZ Density')
        axes[1, 0].set_aspect('equal')
        
        # 3D preview
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, alpha=0.5)
        if mesh_vertices is not None:
            ax3d.scatter(
                mesh_vertices[::10, 0],
                mesh_vertices[::10, 1],
                mesh_vertices[::10, 2],
                c='red',
                s=0.5,
                alpha=0.2,
                label='Mesh'
            )
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('3D View')
        self._set_axes_equal(ax3d)
        
        plt.tight_layout()
        return fig
        
    def save_figure(self, fig: Figure, path: Union[str, Path], dpi: int = 150) -> None:
        """Save figure to file.
        
        Args:
            fig: Matplotlib figure
            path: Output file path
            dpi: Resolution in dots per inch
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
    def _set_axes_equal(self, ax: Axes3D) -> None:
        """Set equal aspect ratio for 3D axes."""
        # Get current limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        # Find range for each axis
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        # Find maximum range
        max_range = max([x_range, y_range, z_range])
        
        # Set new limits
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


# Convenience functions
def plot_point_cloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
    **kwargs
) -> Union[Figure, 'go.Figure', None]:
    """Quick function to plot a point cloud.
    
    Args:
        points: Point cloud array (N, 3)
        title: Plot title
        save_path: Optional path to save figure
        interactive: Use Plotly for interactive plot
        **kwargs: Additional arguments for plotting
        
    Returns:
        Figure object
    """
    viz = PointCloudVisualizer()
    
    if interactive and PLOTLY_AVAILABLE:
        fig = viz.plot_points_interactive(points, title, **kwargs)
        if save_path and fig:
            fig.write_html(str(save_path))
    else:
        fig = viz.plot_points_3d(points, title, **kwargs)
        if save_path:
            viz.save_figure(fig, save_path)
            
    return fig


def save_point_cloud_image(
    points: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Point Cloud",
    views: List[Tuple[float, float]] = [(30, 45), (30, 135), (90, 0)],
    dpi: int = 150,
) -> None:
    """Save point cloud visualizations from multiple views.
    
    Args:
        points: Point cloud array
        output_path: Base path for output (view angle will be appended)
        title: Plot title
        views: List of (elevation, azimuth) viewing angles
        dpi: Output resolution
    """
    viz = PointCloudVisualizer()
    output_path = Path(output_path)
    
    for elev, azim in views:
        fig = viz.plot_points_3d(points, title, elev=elev, azim=azim)
        
        # Create filename with view angles
        stem = output_path.stem
        suffix = output_path.suffix
        view_path = output_path.parent / f"{stem}_elev{elev}_azim{azim}{suffix}"
        
        viz.save_figure(fig, view_path, dpi=dpi)


def compare_sampling_methods(
    results: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Sampling Method Comparison",
) -> Figure:
    """Compare different sampling methods visually.
    
    Args:
        results: Dictionary mapping method names to point clouds
        output_path: Optional path to save comparison
        title: Figure title
        
    Returns:
        Comparison figure
    """
    viz = PointCloudVisualizer(figsize=(15, 5))
    fig = viz.compare_point_clouds(results, title)
    
    if output_path:
        viz.save_figure(fig, output_path)
        
    return fig