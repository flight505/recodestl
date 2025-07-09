"""Command-line interface for RecodeSTL."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from recodestl.core import Config, Converter
from recodestl.processing import load_stl, validate_stl, preprocess_mesh
from recodestl.sampling import SamplingFactory
from recodestl.utils import create_cache_manager, setup_logging

app = typer.Typer(
    name="recodestl",
    help="Convert STL files to parametric CAD models using AI",
    add_completion=False,
)
console = Console()

# Set up structured logging based on config
_logger_setup = False


def _ensure_logging_setup(config: Optional[Config] = None) -> None:
    """Ensure logging is set up once."""
    global _logger_setup
    if not _logger_setup:
        if config is None:
            config = Config()
        setup_logging(config.logging)
        _logger_setup = True


@app.command()
def analyze(
    stl_file: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to STL file to analyze",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed analysis",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate mesh",
    ),
) -> None:
    """Analyze an STL file and display information."""
    console.print(f"\n🔍 Analyzing [cyan]{stl_file.name}[/cyan]...")
    
    try:
        # Create cache manager if needed
        cache_mgr = create_cache_manager() if Config().cache.enabled else None
        
        # Load mesh
        with console.status("Loading STL file..."):
            mesh = load_stl(stl_file, validate=False, show_progress=True, cache_manager=cache_mgr)
            
        # Basic info
        table = Table(title="Mesh Information", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("File", str(stl_file))
        table.add_row("File Size", f"{stl_file.stat().st_size / 1024 / 1024:.2f} MB")
        table.add_row("Vertices", f"{len(mesh.vertices):,}")
        table.add_row("Faces", f"{len(mesh.faces):,}")
        table.add_row("Edges", f"{len(mesh.edges):,}")
        
        if mesh.bounds is not None:
            bounds_min = mesh.bounds[0]
            bounds_max = mesh.bounds[1]
            extents = mesh.extents
            table.add_row(
                "Bounding Box",
                f"[{bounds_min[0]:.2f}, {bounds_min[1]:.2f}, {bounds_min[2]:.2f}] to "
                f"[{bounds_max[0]:.2f}, {bounds_max[1]:.2f}, {bounds_max[2]:.2f}]"
            )
            table.add_row(
                "Size",
                f"{extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}"
            )
            
        console.print(table)
        
        # Validation
        if validate:
            console.print("\n📋 Validation Results:")
            report = validate_stl(stl_file)
            
            # Status
            if report.is_valid:
                console.print("  ✅ Mesh is valid")
            else:
                console.print("  ❌ Mesh has issues:")
                for error in report.errors:
                    console.print(f"     • {error}", style="red")
                    
            if report.warnings:
                console.print("  ⚠️  Warnings:")
                for warning in report.warnings:
                    console.print(f"     • {warning}", style="yellow")
                    
            # Properties
            console.print(f"\n  • Watertight: {'✅' if report.is_watertight else '❌'}")
            if report.volume is not None:
                console.print(f"  • Volume: {report.volume:.3f} units³")
            console.print(f"  • Surface Area: {report.surface_area:.3f} units²")
            
            # Topology
            if detailed:
                console.print("\n📐 Topology:")
                console.print(f"  • Connected Components: {report.connected_components}")
                console.print(f"  • Euler Characteristic: {report.euler_characteristic}")
                if report.is_watertight:
                    console.print(f"  • Genus: {report.genus}")
                if report.has_holes:
                    console.print(f"  • Holes: {report.hole_count}")
                    
            # Quality metrics
            if detailed:
                console.print("\n📊 Quality Metrics:")
                console.print(f"  • Face Areas: min={report.face_area_min:.6f}, "
                             f"max={report.face_area_max:.6f}, "
                             f"mean={report.face_area_mean:.6f}")
                console.print(f"  • Edge Lengths: min={report.edge_length_min:.6f}, "
                             f"max={report.edge_length_max:.6f}, "
                             f"mean={report.edge_length_mean:.6f}")
                             
            # Features
            if report.features:
                console.print("\n🔧 Detected Features:")
                for feature in report.features:
                    console.print(f"  • {feature}")
                    
            # Recommendations
            console.print("\n🎯 Recommendations:")
            console.print(f"  • Sampling Method: [green]{report.recommended_method}[/green]")
            console.print(f"  • Point Count: [green]{report.recommended_points}[/green]")
            if report.curvature_percentage > 0:
                console.print(f"  • High Curvature: {report.curvature_percentage:.1f}% of vertices")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sample(
    stl_file: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to STL file",
    ),
    num_points: int = typer.Option(
        256,
        "--points",
        "-n",
        help="Number of points to sample",
    ),
    method: str = typer.Option(
        "uniform",
        "--method",
        "-m",
        help="Sampling method",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for point cloud (numpy format)",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Show preview of sampled points",
    ),
    visualize: Optional[Path] = typer.Option(
        None,
        "--visualize",
        "-v",
        help="Save visualization to file",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Create interactive HTML visualization",
    ),
) -> None:
    """Sample point cloud from STL file."""
    console.print(f"\n🎯 Sampling points from [cyan]{stl_file.name}[/cyan]...")
    
    try:
        # Create cache manager if needed
        cache_mgr = create_cache_manager() if Config().cache.enabled else None
        
        # Load mesh
        with console.status("Loading STL file..."):
            mesh = load_stl(stl_file, show_progress=True, cache_manager=cache_mgr)
            
        # Preprocess mesh
        with console.status("Preprocessing mesh..."):
            mesh, transform_info = preprocess_mesh(mesh)
            
        # Create sampler
        sampler = SamplingFactory.create(method, num_points=num_points, cache_manager=cache_mgr)
        
        # Sample points
        with console.status(f"Sampling {num_points} points using {method} method..."):
            points = sampler.sample(mesh)
            
        console.print(f"✅ Sampled {len(points)} points")
        
        # Save if requested
        if output:
            import numpy as np
            np.save(output, points)
            console.print(f"💾 Saved point cloud to [cyan]{output}[/cyan]")
            
        # Preview if requested
        if preview:
            console.print("\n📊 Point Cloud Statistics:")
            console.print(f"  • Shape: {points.shape}")
            console.print(f"  • Min: [{points.min(axis=0)[0]:.3f}, "
                         f"{points.min(axis=0)[1]:.3f}, {points.min(axis=0)[2]:.3f}]")
            console.print(f"  • Max: [{points.max(axis=0)[0]:.3f}, "
                         f"{points.max(axis=0)[1]:.3f}, {points.max(axis=0)[2]:.3f}]")
            console.print(f"  • Mean: [{points.mean(axis=0)[0]:.3f}, "
                         f"{points.mean(axis=0)[1]:.3f}, {points.mean(axis=0)[2]:.3f}]")
                         
        # Visualize if requested
        if visualize:
            from recodestl.visualization import plot_point_cloud
            
            console.print(f"\n🎨 Creating visualization...")
            
            if interactive:
                # Save as interactive HTML
                viz_path = visualize.with_suffix(".html")
                plot_point_cloud(
                    points,
                    title=f"{stl_file.stem} - {method} sampling ({num_points} points)",
                    save_path=viz_path,
                    interactive=True,
                )
                console.print(f"💾 Saved interactive visualization to [cyan]{viz_path}[/cyan]")
            else:
                # Save as static image
                plot_point_cloud(
                    points,
                    title=f"{stl_file.stem} - {method} sampling ({num_points} points)",
                    save_path=visualize,
                    interactive=False,
                )
                console.print(f"💾 Saved visualization to [cyan]{visualize}[/cyan]")
                         
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def convert(
    stl_files: List[Path] = typer.Argument(
        ...,
        exists=True,
        help="STL files to convert",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for STEP files",
    ),
    method: str = typer.Option(
        "adaptive",
        "--method",
        "-m",
        help="Sampling method",
    ),
    num_points: int = typer.Option(
        256,
        "--points",
        "-n",
        help="Number of points to sample",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        "-p",
        help="Process files in parallel",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (cpu, cuda, mps)",
    ),
    use_mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock model for testing",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
) -> None:
    """Convert STL files to parametric STEP models."""
    console.print(f"\n🚀 Converting {len(stl_files)} STL file(s)...")
    
    try:
        # Load configuration
        if config:
            cfg = Config.from_toml(config)
        else:
            cfg = Config()
            
        # Set up logging with specified level
        from recodestl.core.config import LoggingConfig
        log_config = cfg.logging.model_dump()
        log_config["level"] = log_level.upper()
        cfg = Config(
            sampling=cfg.sampling,
            model=cfg.model,
            export=cfg.export,
            cache=cfg.cache,
            processing=cfg.processing,
            logging=LoggingConfig(**log_config),
            feature_detection=cfg.feature_detection,
        )
        _ensure_logging_setup(cfg)
            
        # Override with command line options by creating new config
        from recodestl.core.config import SamplingConfig, ModelConfig
        
        sampling_kwargs = cfg.sampling.model_dump()
        sampling_kwargs["method"] = method
        sampling_kwargs["num_points"] = num_points
        
        model_kwargs = cfg.model.model_dump()
        if device:
            model_kwargs["device"] = device
            
        cfg = Config(
            sampling=SamplingConfig(**sampling_kwargs),
            model=ModelConfig(**model_kwargs),
            export=cfg.export,
            cache=cfg.cache,
            processing=cfg.processing,
            logging=cfg.logging,
            feature_detection=cfg.feature_detection,
        )
            
        # Create output directory if specified (and it's a directory)
        if output_dir and not str(output_dir).endswith('.step'):
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create converter
        converter = Converter(config=cfg, console=console)
        
        # Use mock model if requested
        if use_mock:
            from recodestl.models.mock_model import MockCADRecodeModel
            converter.model = MockCADRecodeModel(
                device=cfg.model.device,
                cache_manager=converter.cache_manager,
            )
            
        # Load model
        with console.status("Loading model..."):
            converter.load_model()
            
        # Show device info
        device_info = converter.get_device_info()
        console.print(f"Using device: [cyan]{device_info['config']['device']}[/cyan]")
        
        # Convert files
        if len(stl_files) == 1 and output_dir and str(output_dir).endswith('.step'):
            # Single file with specific output path
            results = [converter.convert_single(
                stl_files[0],
                output_path=output_dir,
                progress_callback=lambda msg: console.print(f"  {msg}"),
            )]
        else:
            # Batch conversion
            results = converter.convert_batch(
                stl_files,
                output_dir=output_dir,
                parallel=parallel,
                progress_callback=lambda msg: console.print(f"  {msg}"),
            )
        
        # Report results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        console.print(f"\n✅ Successfully converted {successful}/{len(results)} files")
        
        if failed > 0:
            console.print(f"❌ Failed to convert {failed} files:")
            for result in results:
                if not result.success:
                    console.print(f"  • {result.input_path.name}: {result.error}")
                    
        # Show timing statistics
        if successful > 0:
            total_time = sum(r.metrics.get("total_time", 0) for r in results if r.success)
            avg_time = total_time / successful
            console.print(f"\n⏱️  Average conversion time: {avg_time:.2f}s per file")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Cleanup
        if 'converter' in locals():
            converter.cleanup()


@app.command()
def compare(
    stl_file: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to STL file",
    ),
    num_points: int = typer.Option(
        256,
        "--points",
        "-n",
        help="Number of points to sample",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for comparison image",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Create interactive comparison",
    ),
) -> None:
    """Compare different sampling methods on the same STL file."""
    console.print(f"\n🔬 Comparing sampling methods for [cyan]{stl_file.name}[/cyan]...")
    
    try:
        # Create cache manager if needed
        cache_mgr = create_cache_manager() if Config().cache.enabled else None
        
        # Load mesh
        with console.status("Loading STL file..."):
            mesh = load_stl(stl_file, show_progress=True, cache_manager=cache_mgr)
            
        # Preprocess mesh
        with console.status("Preprocessing mesh..."):
            mesh, transform_info = preprocess_mesh(mesh)
            
        # Sample with each method
        methods = ["uniform", "poisson", "adaptive"]
        results = {}
        
        for method in methods:
            with console.status(f"Sampling with {method} method..."):
                sampler = SamplingFactory.create(method, num_points=num_points, cache_manager=cache_mgr)
                points = sampler.sample(mesh)
                results[method] = points
                console.print(f"  ✅ {method}: {len(points)} points")
                
        # Create comparison visualization
        if output or interactive:
            from recodestl.visualization import compare_sampling_methods
            
            console.print(f"\n🎨 Creating comparison visualization...")
            
            if interactive and output:
                # Save as interactive HTML if output path provided
                output_path = output.with_suffix(".html")
                # For now, we'll save individual interactive plots
                from recodestl.visualization import plot_point_cloud
                for method, points in results.items():
                    method_path = output_path.parent / f"{output_path.stem}_{method}.html"
                    plot_point_cloud(
                        points,
                        title=f"{stl_file.stem} - {method} sampling",
                        save_path=method_path,
                        interactive=True,
                    )
                console.print(f"💾 Saved interactive visualizations to [cyan]{output_path.parent}[/cyan]")
            else:
                # Create static comparison image
                fig = compare_sampling_methods(
                    results,
                    output_path=output,
                    title=f"Sampling Comparison - {stl_file.stem} ({num_points} points)",
                )
                if output:
                    console.print(f"💾 Saved comparison to [cyan]{output}[/cyan]")
                else:
                    # Show the figure if no output specified
                    import matplotlib.pyplot as plt
                    plt.show()
                    
        # Show statistics
        console.print("\n📊 Sampling Statistics:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Points", justify="right")
        table.add_column("Min Z", justify="right")
        table.add_column("Max Z", justify="right")
        table.add_column("Std Dev", justify="right")
        
        for method, points in results.items():
            table.add_row(
                method,
                str(len(points)),
                f"{points[:, 2].min():.3f}",
                f"{points[:, 2].max():.3f}",
                f"{points.std():.3f}",
            )
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Display information about RecodeSTL."""
    console.print("\n[cyan]RecodeSTL[/cyan] - STL to Parametric CAD Converter")
    console.print("Version: 0.1.0")
    console.print("\nFeatures:")
    console.print("  • 🤖 AI-powered CAD code generation")
    console.print("  • 🎯 Multiple sampling strategies")
    console.print("  • 🔧 Mechanical feature preservation")
    console.print("  • 🚀 Apple Silicon optimization")
    console.print("  • 📦 Batch processing support")
    
    # Show available sampling methods
    from recodestl.sampling import SamplingFactory
    methods = SamplingFactory.available_methods()
    console.print(f"\nAvailable sampling methods: {', '.join(methods)}")
    
    # Check device availability
    import torch
    if torch.backends.mps.is_available():
        console.print("\n✅ Metal Performance Shaders (MPS) available")
    elif torch.cuda.is_available():
        console.print("\n✅ CUDA available")
    else:
        console.print("\n⚠️  No GPU acceleration available (using CPU)")


@app.command()
def cache(
    action: str = typer.Argument(
        ...,
        help="Cache action: stats, clear, or evict"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file"
    ),
) -> None:
    """Manage the cache system."""
    # Load configuration
    if config:
        cfg = Config.from_toml(config)
    else:
        cfg = Config()
        
    # Create cache manager
    cache_mgr = create_cache_manager(cfg.cache)
    
    if action == "stats":
        # Show cache statistics
        stats = cache_mgr.get_stats()
        
        if not stats.get("enabled", False):
            console.print("[yellow]Cache is disabled[/yellow]")
            return
            
        table = Table(title="Cache Statistics", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Location", stats.get("location", "N/A"))
        table.add_row("Size Limit", f"{stats.get('size_limit_gb', 0):.1f} GB")
        table.add_row("Entries", f"{stats.get('entries', 0):,}")
        table.add_row("Size", f"{stats.get('size_mb', 0):.1f} MB")
        table.add_row("Hits", f"{stats.get('hits', 0):,}")
        table.add_row("Misses", f"{stats.get('misses', 0):,}")
        table.add_row("Hit Rate", f"{stats.get('hit_rate', 0):.1%}")
        
        console.print(table)
        
    elif action == "clear":
        # Clear cache with confirmation
        if typer.confirm("Are you sure you want to clear the cache?"):
            cache_mgr.clear()
            console.print("[green]Cache cleared successfully[/green]")
        else:
            console.print("[yellow]Cache clear cancelled[/yellow]")
            
    elif action == "evict":
        # Evict expired entries
        count = cache_mgr.evict_expired()
        console.print(f"[green]Evicted {count} expired entries[/green]")
        
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: stats, clear, evict")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()