#!/usr/bin/env python3
"""Test the converter pipeline with mock model."""

from pathlib import Path

from recodestl.core import Config, Converter
from recodestl.models.mock_model import MockCADRecodeModel


def main():
    """Test converter with Sonos wall mount."""
    stl_file = Path("tests/fixtures/Sonos_play_1_wall.stl")
    output_file = Path("tests/output/sonos_converted.step")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    print("Testing converter pipeline...")
    print(f"Input: {stl_file}")
    print(f"Output: {output_file}")
    
    # Load model
    converter.load_model()
    
    # Convert
    result = converter.convert_single(
        stl_file,
        output_file,
        progress_callback=lambda msg: print(f"  {msg}")
    )
    
    # Report results
    if result.success:
        print(f"\n✅ Conversion successful!")
        print(f"  Output: {result.output_path}")
        print(f"  Time: {result.metrics['total_time']:.2f}s")
        print("\nBreakdown:")
        for key, value in result.metrics.items():
            if key.endswith("_time"):
                print(f"  {key}: {value:.3f}s")
    else:
        print(f"\n❌ Conversion failed: {result.error}")
    
    # Cleanup
    converter.cleanup()


if __name__ == "__main__":
    main()