"""Command-line interface for RecodeSTL."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from recodestl.core import Config
from recodestl.processing import load_stl, validate_stl, preprocess_mesh
from recodestl.sampling import SamplingFactory

app = typer.Typer(
    name="recodestl",
    help="Convert STL files to parametric CAD models using AI",
    add_completion=False,
)
console = Console()


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
        # Load mesh
        with console.status("Loading STL file..."):
            mesh = load_stl(stl_file, validate=False, show_progress=True)
            
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
) -> None:
    """Sample point cloud from STL file."""
    console.print(f"\n🎯 Sampling points from [cyan]{stl_file.name}[/cyan]...")
    
    try:
        # Load mesh
        with console.status("Loading STL file..."):
            mesh = load_stl(stl_file, show_progress=True)
            
        # Preprocess mesh
        with console.status("Preprocessing mesh..."):
            mesh, transform_info = preprocess_mesh(mesh)
            
        # Create sampler
        sampler = SamplingFactory.create(method, num_points=num_points)
        
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


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()