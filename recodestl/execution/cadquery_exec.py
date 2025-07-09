"""CadQuery code execution and STEP export functionality."""

import tempfile
from pathlib import Path
from typing import Any, Optional

import cadquery as cq

from recodestl.core.exceptions import CodeExecutionError, ExportError
from recodestl.execution.sandbox import SecureExecutor


class CadQueryExecutor:
    """Executes CadQuery code and manages CAD model export."""

    def __init__(self, timeout: int = 30):
        """Initialize CadQuery executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.executor = SecureExecutor(timeout=timeout)

    def execute_code(self, code: str) -> cq.Workplane:
        """Execute CadQuery code and return the resulting model.

        Args:
            code: CadQuery Python code

        Returns:
            CadQuery Workplane object

        Raises:
            CodeValidationError: If code validation fails
            CodeExecutionError: If code execution fails
            TimeoutError: If execution times out
        """
        # Try to find the result variable name in the code
        result_vars = ["result", "wp", "workplane", "model", "part"]

        for var_name in result_vars:
            if var_name in code:
                try:
                    result = self.executor.execute(code, result_var=var_name)
                    if isinstance(result, cq.Workplane):
                        return result
                except CodeExecutionError:
                    continue

        # If no known variable found, try default execution
        result = self.executor.execute(code, result_var="result")

        if not isinstance(result, cq.Workplane):
            raise CodeExecutionError(
                code,
                f"No CadQuery Workplane found in execution result",
            )

        return result

    def validate_workplane(self, workplane: cq.Workplane) -> tuple[bool, list[str]]:
        """Validate a CadQuery Workplane.

        Args:
            workplane: CadQuery Workplane to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Check if workplane has any solids
            if not workplane.solids().size():
                errors.append("No solids found in workplane")

            # Check if the model is valid
            for solid in workplane.solids().all():
                if not solid.isValid():
                    errors.append("Invalid solid geometry detected")

                # Check volume
                volume = solid.Volume()
                if volume <= 0:
                    errors.append(f"Invalid volume: {volume}")

                # Check for self-intersections
                if not solid.fix():
                    errors.append("Failed to fix solid geometry")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors

    def export_step(
        self,
        workplane: cq.Workplane,
        output_path: Path,
        precision: float = 0.001,
        angular_tolerance: float = 0.1,
    ) -> None:
        """Export CadQuery Workplane to STEP file.

        Args:
            workplane: CadQuery Workplane to export
            output_path: Path for output STEP file
            precision: Linear precision for export
            angular_tolerance: Angular tolerance in degrees

        Raises:
            ExportError: If export fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate workplane before export
            is_valid, errors = self.validate_workplane(workplane)
            if not is_valid:
                raise ExportError(output_path, f"Invalid model: {', '.join(errors)}")

            # Export to STEP with specified precision
            workplane.val().exportStep(
                str(output_path),
                tolerance=precision,
                angularTolerance=angular_tolerance,
            )

            # Verify file was created
            if not output_path.exists():
                raise ExportError(output_path, "STEP file was not created")

            # Check file size
            if output_path.stat().st_size == 0:
                raise ExportError(output_path, "STEP file is empty")

        except ExportError:
            raise
        except Exception as e:
            raise ExportError(output_path, str(e))

    def export_stl(
        self,
        workplane: cq.Workplane,
        output_path: Path,
        linear_tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
    ) -> None:
        """Export CadQuery Workplane to STL file.

        Args:
            workplane: CadQuery Workplane to export
            output_path: Path for output STL file
            linear_tolerance: Linear tolerance for tessellation
            angular_tolerance: Angular tolerance in degrees

        Raises:
            ExportError: If export fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export to STL
            workplane.val().exportStl(
                str(output_path),
                tolerance=linear_tolerance,
                angularTolerance=angular_tolerance,
            )

            # Verify file was created
            if not output_path.exists():
                raise ExportError(output_path, "STL file was not created")

        except Exception as e:
            raise ExportError(output_path, str(e))

    def execute_and_export(
        self,
        code: str,
        output_path: Path,
        export_format: str = "step",
        precision: float = 0.001,
        angular_tolerance: float = 0.1,
        validate: bool = True,
    ) -> cq.Workplane:
        """Execute CadQuery code and export the result.

        Args:
            code: CadQuery Python code
            output_path: Path for output file
            export_format: Export format ("step" or "stl")
            precision: Linear precision for export
            angular_tolerance: Angular tolerance in degrees
            validate: Whether to validate before export

        Returns:
            CadQuery Workplane object

        Raises:
            CodeValidationError: If code validation fails
            CodeExecutionError: If code execution fails
            ExportError: If export fails
            TimeoutError: If execution times out
        """
        # Execute code
        workplane = self.execute_code(code)

        # Validate if requested
        if validate:
            is_valid, errors = self.validate_workplane(workplane)
            if not is_valid:
                raise CodeExecutionError(
                    code,
                    f"Generated invalid model: {', '.join(errors)}",
                )

        # Export based on format
        if export_format.lower() == "step":
            self.export_step(
                workplane,
                output_path,
                precision=precision,
                angular_tolerance=angular_tolerance,
            )
        elif export_format.lower() == "stl":
            self.export_stl(
                workplane,
                output_path,
                linear_tolerance=precision,
                angular_tolerance=angular_tolerance,
            )
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return workplane

    def preview_code(self, code: str) -> dict[str, Any]:
        """Execute code and return preview information.

        Args:
            code: CadQuery Python code

        Returns:
            Dictionary with preview information

        Raises:
            CodeValidationError: If code validation fails
            CodeExecutionError: If code execution fails
        """
        workplane = self.execute_code(code)

        # Get model information
        solids = workplane.solids()
        faces = workplane.faces()
        edges = workplane.edges()
        vertices = workplane.vertices()

        # Calculate bounding box
        bb = workplane.val().BoundingBox()

        preview_info = {
            "num_solids": solids.size(),
            "num_faces": faces.size(),
            "num_edges": edges.size(),
            "num_vertices": vertices.size(),
            "bounding_box": {
                "min": (bb.xmin, bb.ymin, bb.zmin),
                "max": (bb.xmax, bb.ymax, bb.zmax),
                "size": (bb.xlen, bb.ylen, bb.zlen),
            },
            "volume": sum(s.Volume() for s in solids.all()),
            "is_valid": all(s.isValid() for s in solids.all()),
        }

        return preview_info


def create_executor(timeout: int = 30) -> CadQueryExecutor:
    """Create a CadQuery executor instance.

    Args:
        timeout: Maximum execution time in seconds

    Returns:
        CadQueryExecutor instance
    """
    return CadQueryExecutor(timeout=timeout)