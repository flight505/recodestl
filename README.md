# RecodeSTL

Convert STL files to parametric CAD models using the CAD-Recode neural model.

## Features

- **AI-Powered**: Uses CAD-Recode v1.5 (1.5B parameter model) to generate parametric CAD code
- **Smart Sampling**: Multiple point cloud sampling strategies (adaptive, poisson, uniform)
- **Feature Preservation**: Maintains mechanical features like threads, fillets, and chamfers
- **Apple Silicon Optimized**: Native support for M-series Macs with Metal Performance Shaders
- **Batch Processing**: Convert multiple files with parallel processing
- **Caching**: Smart caching for faster repeated conversions

## Installation

### Prerequisites

- Python 3.11 or higher
- macOS (optimized for Apple Silicon) or Linux
- 64GB RAM recommended for optimal performance

### Install with uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/recodestl.git
cd recodestl

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Install with pip

```bash
pip install recodestl
```

## Quick Start

### Convert a single STL file

```bash
recodestl convert input.stl -o output/
```

### Analyze an STL file

```bash
recodestl analyze model.stl --detailed
```

### Batch conversion with parallel processing

```bash
recodestl convert *.stl --parallel -o output/
```

### Use custom configuration

```bash
recodestl --config config.toml convert input.stl
```

## Configuration

Create a `config.toml` file to customize settings:

```toml
[model]
device = "mps"  # Use "cuda" for NVIDIA GPUs, "cpu" for CPU-only
max_tokens = 768
temperature = 0.0

[sampling]
num_points = 256
method = "adaptive"  # Options: adaptive, poisson, uniform
curvature_weight = 0.7

[processing]
parallel_enabled = true
batch_size = 4
timeout = 60

[cache]
enabled = true
max_size_gb = 20.0
```

## Development

### Setup development environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
ruff format .

# Type checking
mypy recodestl
```

## Architecture

RecodeSTL follows a modular architecture:

- **CLI**: Typer-based command-line interface
- **Core**: Main conversion pipeline and configuration
- **Models**: CAD-Recode model wrapper with device optimization
- **Sampling**: Point cloud generation strategies
- **Execution**: Secure sandboxed code execution
- **Export**: STEP file generation and validation

## Performance

On MacBook M3 Max (64GB):
- Single file conversion: ~10-15 seconds
- Memory usage: ~6-12GB for model + 2-4GB per file
- Batch processing: Linear scaling with parallel execution

## License

MIT License - see LICENSE file for details

## Acknowledgments

- CAD-Recode model by [filaPro](https://github.com/filaPro/cad-recode)
- Built with CadQuery for parametric CAD generation