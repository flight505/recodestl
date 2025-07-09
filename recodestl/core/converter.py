"""Main converter pipeline for STL to parametric CAD conversion."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from recodestl.core.config import Config
from recodestl.core.exceptions import (
    CodeExecutionError,
    ConverterError,
    ExportError,
    ModelError,
    PointCloudError,
    STLLoadError,
)
from recodestl.execution import CadQueryExecutor, SecureExecutor
from recodestl.models import CADRecodeModel, create_model
from recodestl.processing import load_stl, preprocess_mesh, validate_stl
from recodestl.sampling import SamplingFactory
from recodestl.utils import CacheManager, create_cache_manager

logger = logging.getLogger(__name__)


class ConversionResult:
    """Result of a conversion operation."""

    def __init__(
        self,
        success: bool,
        input_path: Path,
        output_path: Optional[Path] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Initialize conversion result.

        Args:
            success: Whether conversion succeeded
            input_path: Input STL file path
            output_path: Output STEP file path (if successful)
            error: Error message (if failed)
            metrics: Performance and quality metrics
        """
        self.success = success
        self.input_path = input_path
        self.output_path = output_path
        self.error = error
        self.metrics = metrics or {}
        self.timestamp = time.time()


class Converter:
    """Main converter for STL to parametric CAD conversion."""

    def __init__(
        self,
        config: Optional[Config] = None,
        cache_manager: Optional[CacheManager] = None,
        model: Optional[CADRecodeModel] = None,
        console: Optional[Console] = None,
    ):
        """Initialize converter.

        Args:
            config: Configuration object
            cache_manager: Cache manager (will create if not provided)
            model: CAD-Recode model (will create if not provided)
            console: Rich console for output
        """
        self.config = config or Config()
        self.cache_manager = cache_manager or (
            create_cache_manager(self.config.cache) if self.config.cache.enabled else None
        )
        self.model = model
        self.console = console or Console()
        
        # Create executors
        self.secure_executor = SecureExecutor(
            timeout=self.config.processing.timeout,
        )
        self.cadquery_executor = CadQueryExecutor(
            secure_executor=self.secure_executor,
            cache_manager=self.cache_manager,
        )
        
        self._model_loaded = False

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load the CAD-Recode model.

        Args:
            model_path: Optional path to model weights

        Raises:
            ModelError: If model loading fails
        """
        if self._model_loaded and self.model is not None:
            return
            
        try:
            if self.model is None:
                self.model = create_model(
                    device=self.config.model.device,
                    dtype=self.config.model.dtype,
                    cache_manager=self.cache_manager,
                    max_new_tokens=self.config.model.max_tokens,
                    temperature=self.config.model.temperature,
                )
            
            self.model.load_model(cache_dir=model_path)
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")

    def convert_single(
        self,
        stl_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ConversionResult:
        """Convert a single STL file to parametric CAD.

        Args:
            stl_path: Path to input STL file
            output_path: Path for output STEP file (auto-generated if None)
            progress_callback: Optional callback for progress updates

        Returns:
            ConversionResult with success status and metrics
        """
        stl_path = Path(stl_path)
        start_time = time.time()
        metrics = {}
        
        try:
            # Validate input
            if not stl_path.exists():
                raise STLLoadError(stl_path, "File does not exist")
                
            # Auto-generate output path if needed
            if output_path is None:
                output_path = stl_path.with_suffix(".step")
            else:
                output_path = Path(output_path)
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. Load and validate mesh
            if progress_callback:
                progress_callback("Loading STL file...")
                
            mesh = load_stl(
                stl_path,
                validate=self.config.processing.validate_input,
                show_progress=False,
                cache_manager=self.cache_manager,
            )
            metrics["mesh_load_time"] = time.time() - start_time
            metrics["vertex_count"] = len(mesh.vertices)
            metrics["face_count"] = len(mesh.faces)
            
            # 2. Preprocess mesh
            if progress_callback:
                progress_callback("Preprocessing mesh...")
                
            mesh, transform_info = preprocess_mesh(
                mesh,
                repair=self.config.processing.repair_mesh,
                simplify=self.config.processing.simplify_mesh,
                target_faces=self.config.processing.simplify_target_faces,
            )
            metrics["preprocess_time"] = time.time() - start_time - metrics["mesh_load_time"]
            
            # 3. Generate point cloud
            if progress_callback:
                progress_callback(f"Sampling {self.config.sampling.num_points} points...")
                
            sampler = SamplingFactory.create(
                self.config.sampling.method,
                num_points=self.config.sampling.num_points,
                cache_manager=self.cache_manager,
                **self.config.sampling.method_params,
            )
            
            point_cloud = sampler.sample(mesh)
            metrics["sampling_time"] = (
                time.time() - start_time - metrics["mesh_load_time"] - metrics["preprocess_time"]
            )
            
            # 4. Run model inference
            if progress_callback:
                progress_callback("Generating CAD code...")
                
            if not self._model_loaded:
                self.load_model()
                
            cad_code = self.model.generate(
                point_cloud,
                max_new_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                do_sample=self.config.model.temperature > 0,
            )
            metrics["inference_time"] = (
                time.time()
                - start_time
                - metrics["mesh_load_time"]
                - metrics["preprocess_time"]
                - metrics["sampling_time"]
            )
            
            # 5. Execute code and export
            if progress_callback:
                progress_callback("Executing CAD code and exporting...")
                
            step_file = self.cadquery_executor.execute_and_export(
                cad_code,
                output_path,
                file_format=self.config.export.file_format,
                precision=self.config.export.precision,
                assembly_mode=self.config.export.assembly_mode,
            )
            
            metrics["export_time"] = (
                time.time()
                - start_time
                - metrics["mesh_load_time"]
                - metrics["preprocess_time"]
                - metrics["sampling_time"]
                - metrics["inference_time"]
            )
            metrics["total_time"] = time.time() - start_time
            metrics["output_size"] = step_file.stat().st_size
            
            # Log success
            logger.info(
                f"Successfully converted {stl_path.name} to {output_path.name} "
                f"in {metrics['total_time']:.2f}s"
            )
            
            return ConversionResult(
                success=True,
                input_path=stl_path,
                output_path=output_path,
                metrics=metrics,
            )
            
        except Exception as e:
            # Log error
            error_msg = f"Failed to convert {stl_path.name}: {str(e)}"
            logger.error(error_msg)
            
            return ConversionResult(
                success=False,
                input_path=stl_path,
                error=error_msg,
                metrics=metrics,
            )

    def convert_batch(
        self,
        stl_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[ConversionResult]:
        """Convert multiple STL files to parametric CAD.

        Args:
            stl_paths: List of STL file paths
            output_dir: Output directory (uses input dirs if None)
            parallel: Whether to process in parallel
            max_workers: Maximum parallel workers (auto if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List of ConversionResult objects
        """
        results = []
        
        # Ensure model is loaded before parallel processing
        if parallel and not self._model_loaded:
            self.load_model()
        
        if parallel and len(stl_paths) > 1:
            # Parallel processing
            import concurrent.futures
            from multiprocessing import cpu_count
            
            if max_workers is None:
                max_workers = min(cpu_count(), len(stl_paths), 4)
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = []
                for stl_path in stl_paths:
                    output_path = None
                    if output_dir:
                        output_path = Path(output_dir) / Path(stl_path).with_suffix(".step").name
                        
                    future = executor.submit(
                        self.convert_single,
                        stl_path,
                        output_path,
                        None,  # No progress callback in parallel mode
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    if progress_callback:
                        progress_callback(f"Completed {i+1}/{len(stl_paths)} files")
                    results.append(future.result())
                    
        else:
            # Sequential processing
            for i, stl_path in enumerate(stl_paths):
                if progress_callback:
                    progress_callback(f"Processing file {i+1}/{len(stl_paths)}: {Path(stl_path).name}")
                    
                output_path = None
                if output_dir:
                    output_path = Path(output_dir) / Path(stl_path).with_suffix(".step").name
                    
                result = self.convert_single(stl_path, output_path, progress_callback)
                results.append(result)
                
        return results

    def validate_output(
        self,
        step_path: Union[str, Path],
        original_stl: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Validate generated STEP file.

        Args:
            step_path: Path to STEP file
            original_stl: Optional original STL for comparison

        Returns:
            Validation report dictionary
        """
        # This would validate the STEP file and optionally compare with original
        # For now, just check if file exists and has content
        step_path = Path(step_path)
        
        report = {
            "exists": step_path.exists(),
            "size": step_path.stat().st_size if step_path.exists() else 0,
            "valid": False,
        }
        
        if report["exists"] and report["size"] > 0:
            # Basic validation - check if it's a valid STEP file
            try:
                with open(step_path, "r") as f:
                    header = f.read(100)
                    report["valid"] = "ISO-10303-21" in header
            except Exception:
                pass
                
        return report

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device configuration.

        Returns:
            Device information dictionary
        """
        info = {
            "config": {
                "device": self.config.model.device,
                "dtype": self.config.model.dtype,
                "max_tokens": self.config.model.max_tokens,
            }
        }
        
        if self.model and self._model_loaded:
            info["model"] = self.model.get_device_info()
            
        return info

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            self.model.unload_model()
            self._model_loaded = False
            
        logger.info("Converter cleaned up")