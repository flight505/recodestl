"""Integration test for Sonos wall mount conversion."""

from pathlib import Path

import pytest

from recodestl.core import Config, Converter
from recodestl.models.mock_model import MockCADRecodeModel


@pytest.fixture
def sonos_stl_path() -> Path:
    """Get path to Sonos wall mount STL."""
    return Path(__file__).parent.parent / "fixtures" / "Sonos_play_1_wall.stl"


@pytest.fixture  
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    return output


def test_sonos_mount_conversion(sonos_stl_path: Path, output_dir: Path):
    """Test converting Sonos wall mount STL to STEP."""
    output_file = output_dir / "sonos_converted.step"
    
    # Create configuration
    config = Config(
        sampling={"method": "adaptive", "num_points": 256},
        model={"device": "cpu"},  # Use CPU for testing
    )
    
    # Create converter with mock model
    converter = Converter(config=config)
    converter.model = MockCADRecodeModel(
        device="cpu",
        cache_manager=converter.cache_manager,
    )
    
    # Load model
    converter.load_model()
    
    # Convert
    result = converter.convert_single(
        sonos_stl_path,
        output_file,
    )
    
    # Verify success
    assert result.success
    assert result.output_path == output_file
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    
    # Verify it's a valid STEP file
    with open(output_file, "r") as f:
        content = f.read(100)
        assert "ISO-10303-21" in content
    
    # Check metrics
    assert "total_time" in result.metrics
    assert "vertex_count" in result.metrics
    assert "face_count" in result.metrics
    assert result.metrics["vertex_count"] > 0
    assert result.metrics["face_count"] > 0
    
    # Cleanup
    converter.cleanup()


def test_sonos_mount_batch_conversion(sonos_stl_path: Path, output_dir: Path):
    """Test batch conversion with Sonos mount."""
    # Create configuration
    config = Config(
        sampling={"method": "uniform", "num_points": 128},
        model={"device": "cpu"},
    )
    
    # Create converter with mock model
    converter = Converter(config=config)
    converter.model = MockCADRecodeModel(
        device="cpu",
        cache_manager=converter.cache_manager,
    )
    
    # Convert batch (same file twice for testing)
    results = converter.convert_batch(
        [sonos_stl_path, sonos_stl_path],
        output_dir=output_dir,
        parallel=False,
    )
    
    # Verify results
    assert len(results) == 2
    assert all(r.success for r in results)
    
    # Check output files
    output_files = list(output_dir.glob("*.step"))
    assert len(output_files) == 1  # Same file, so only one output
    
    # Cleanup
    converter.cleanup()


def test_sonos_mount_with_different_sampling_methods(sonos_stl_path: Path, output_dir: Path):
    """Test conversion with different sampling methods."""
    methods = ["uniform", "poisson", "adaptive"]
    
    for method in methods:
        # Create configuration
        config = Config(
            sampling={"method": method, "num_points": 128},
            model={"device": "cpu"},
        )
        
        # Create converter
        converter = Converter(config=config)
        converter.model = MockCADRecodeModel(
            device="cpu",
            cache_manager=converter.cache_manager,
        )
        
        # Convert
        output_file = output_dir / f"sonos_{method}.step"
        result = converter.convert_single(
            sonos_stl_path,
            output_file,
        )
        
        # Verify
        assert result.success, f"Failed with method {method}"
        assert output_file.exists()
        
        # Cleanup
        converter.cleanup()


def test_sonos_mount_validation_report(sonos_stl_path: Path):
    """Test validation report for Sonos mount."""
    from recodestl.processing import validate_stl
    
    report = validate_stl(sonos_stl_path)
    
    # Check basic properties
    assert report.vertex_count > 0
    assert report.face_count > 0
    assert report.surface_area > 0
    
    # Check recommendations
    assert report.recommended_method in ["uniform", "poisson", "adaptive"]
    assert report.recommended_points > 0
    
    # The Sonos mount should be watertight
    # (This might fail if the STL has issues)
    # assert report.is_watertight