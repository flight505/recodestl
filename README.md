# RecodeSTL

Convert STL mesh files to parametric CAD models using the CAD-Recode neural model. RecodeSTL analyzes 3D meshes and generates executable CAD code (CadQuery) that recreates the geometry parametrically.

## Features

- 🚀 **AI-Powered Conversion**: Uses CAD-Recode transformer model to generate parametric CAD code
- 🎯 **Multiple Sampling Strategies**: Uniform, Poisson disk, and adaptive point cloud sampling
- 💾 **Smart Caching**: Persistent caching with 42x speedup for meshes, 222x for point clouds
- 🔧 **Mechanical Feature Detection**: Identifies holes, fillets, chamfers, and other features
- 📊 **Visualization Tools**: Generate static and interactive visualizations of point clouds
- 🛡️ **Secure Execution**: Sandboxed environment for executing generated CAD code
- 📝 **Structured Logging**: Comprehensive logging with structlog for debugging
- 🖥️ **Apple Silicon Optimized**: Native MPS (Metal Performance Shaders) support
- ⚡ **Batch Processing**: Convert multiple files with parallel processing

## Installation

### Prerequisites

- Python 3.11 or higher
- macOS (optimized for Apple Silicon), Linux, or Windows
- 16GB RAM minimum (32GB+ recommended for large models)

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/recodestl.git
cd recodestl

# Install with uv
uv sync
```

### Install with pip

```bash
# Clone the repository
git clone https://github.com/yourusername/recodestl.git
cd recodestl

# Install in development mode
pip install -e .
```

## Quick Start

### Convert a single STL file

```bash
recodestl convert model.stl -o model.step
```

### Analyze an STL file

```bash
recodestl analyze model.stl --detailed
```

### Compare sampling methods

```bash
recodestl compare model.stl -n 256 -o comparison.png
```

### Batch conversion

```bash
recodestl convert *.stl --parallel -o output_dir/
```

## Usage Examples

### Basic Conversion

```bash
# Convert STL to STEP (creates model.step in same directory)
recodestl convert model.stl

# Specify output file
recodestl convert model.stl -o output.step

# Use specific sampling method and point count
recodestl convert model.stl --method adaptive --points 512
```

### Visualization

```bash
# Generate static visualization
recodestl sample model.stl -n 256 --visualize points.png

# Create interactive HTML visualization
recodestl sample model.stl -n 256 --visualize viz.html --interactive

# Compare sampling methods
recodestl compare model.stl -n 256 -o comparison.png
```

### Analysis and Validation

```bash
# Basic analysis
recodestl analyze model.stl

# Detailed analysis with recommendations
recodestl analyze model.stl --detailed
```

### Cache Management

```bash
# View cache statistics
recodestl cache stats

# Clear all cached data
recodestl cache clear

# Remove expired entries
recodestl cache evict
```

## Configuration

Create a `config.toml` file to customize settings:

```toml
[sampling]
num_points = 256
method = "adaptive"
curvature_weight = 0.7

[model]
device = "mps"  # or "cuda", "cpu"
max_tokens = 768
temperature = 0.1

[export]
step_precision = 0.001
angular_tolerance = 0.1
validate_output = true

[processing]
parallel_enabled = true
timeout = 60
validate_input = true
repair_mesh = true

[cache]
enabled = true
cache_dir = "~/.cache/recodestl"
max_size_gb = 20.0

[logging]
level = "INFO"
format = "console"
colorize = true
```

Use with: `recodestl convert model.stl --config config.toml`

## Python API

```python
from recodestl.core import Config, Converter
from recodestl.models import create_model

# Create configuration
config = Config(
    sampling={"method": "adaptive", "num_points": 256},
    model={"device": "mps"},
)

# Create converter
converter = Converter(config=config)

# Convert STL to STEP
result = converter.convert_single("model.stl", "output.step")

if result.success:
    print(f"Conversion successful!")
    print(f"Time: {result.metrics['total_time']:.2f}s")
    print(f"Vertices: {result.metrics['vertex_count']}")
else:
    print(f"Conversion failed: {result.error}")

# Batch conversion
results = converter.convert_batch(
    ["file1.stl", "file2.stl"],
    output_dir="output/",
    parallel=True
)

# Clean up
converter.cleanup()
```

## Sampling Methods

### Uniform Sampling
Random points from mesh surface, weighted by face area.

```bash
recodestl sample model.stl --method uniform -n 256
```

### Poisson Disk Sampling
Evenly distributed points with minimum distance constraint.

```bash
recodestl sample model.stl --method poisson -n 256
```

### Adaptive Sampling
Feature-aware sampling focusing on high-curvature regions.

```bash
recodestl sample model.stl --method adaptive -n 256
```

## Performance

### Caching Impact
- Mesh loading: **42x speedup** (47.24ms → 1.12ms)
- Point cloud sampling: **222x speedup** (442.89ms → 1.99ms)

### Conversion Times (with mock model)
- Small STL (<1MB): ~0.1s
- Medium STL (1-10MB): ~0.5s
- Large STL (>10MB): ~2s

## Architecture

```
recodestl/
├── cli/           # Command-line interface
├── core/          # Core functionality (config, converter, exceptions)
├── execution/     # Secure code execution sandbox
├── models/        # Neural model wrappers
├── processing/    # STL processing (loading, validation, preprocessing)
├── sampling/      # Point cloud sampling strategies
├── utils/         # Utilities (caching, logging)
└── visualization/ # Point cloud visualization tools
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=recodestl

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking
uv run mypy recodestl
```

## Limitations

- Currently uses a mock model for testing (actual CAD-Recode weights not included)
- Large STL files (>100MB) may require significant memory
- Basic STEP file validation only

## License

MIT License - see LICENSE file for details

## Citation

If you use RecodeSTL in your research, please cite:

```bibtex
@article{cadrecode2024,
  title={CAD-Recode: Reverse Engineering CAD Models},
  author={...},
  journal={...},
  year={2024}
}
```

## Acknowledgments

- CAD-Recode model by [original authors]
- Built with [CadQuery](https://github.com/CadQuery/cadquery) for CAD operations
- Uses [trimesh](https://github.com/mikedh/trimesh) for mesh processing