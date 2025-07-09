"""Secure sandboxed execution environment for generated CAD code."""

import ast
import signal
import sys
from contextlib import contextmanager
from io import StringIO
from typing import Any, Callable, Optional, TypeVar, Union

import cadquery as cq
import numpy as np

from recodestl.core.exceptions import CodeExecutionError, CodeValidationError, TimeoutError

T = TypeVar("T")


class SecurityValidator:
    """Validates code for security issues using AST analysis."""

    # Allowed modules for import
    ALLOWED_MODULES = {"math", "numpy", "cadquery", "cq"}

    # AST nodes that are forbidden
    FORBIDDEN_AST_NODES = {
        ast.Global,
        ast.Nonlocal,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Yield,
        ast.YieldFrom,
        ast.GeneratorExp,  # Prevent generators
        # ast.ListComp,  # Allow list comprehensions
    }

    # Forbidden function names
    FORBIDDEN_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "raw_input",
        "file",
        "execfile",
        "reload",
        "vars",
        "locals",
        "globals",
        "dir",
        "help",
        "type",
        "id",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "callable",
    }

    # Forbidden attribute access patterns
    FORBIDDEN_ATTRS = {
        "__",  # Double underscore attributes
        "func_",  # Function internals
        "im_",  # Method internals
        "tb_",  # Traceback internals
        "f_",  # Frame internals
        "gi_",  # Generator internals
    }

    def validate(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate code for security issues.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for forbidden nodes
        for node in ast.walk(tree):
            # Check node type
            if type(node) in self.FORBIDDEN_AST_NODES:
                return False, f"Forbidden construct: {type(node).__name__}"

            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not self._validate_import(node):
                    module = getattr(node, "module", None) or node.names[0].name
                    return False, f"Forbidden import: {module}"

            # Check function calls
            if isinstance(node, ast.Call):
                if not self._validate_call(node):
                    func_name = self._get_call_name(node)
                    return False, f"Forbidden function call: {func_name}"

            # Check attribute access
            if isinstance(node, ast.Attribute):
                if not self._validate_attribute(node):
                    return False, f"Forbidden attribute access: {node.attr}"

        return True, None

    def _validate_import(self, node: Union[ast.Import, ast.ImportFrom]) -> bool:
        """Validate import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in self.ALLOWED_MODULES:
                    return False
        else:  # ast.ImportFrom
            if node.module and node.module not in self.ALLOWED_MODULES:
                return False
        return True

    def _validate_call(self, node: ast.Call) -> bool:
        """Validate function calls."""
        func_name = self._get_call_name(node)
        if func_name in self.FORBIDDEN_FUNCTIONS:
            return False
        return True

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return "unknown"

    def _validate_attribute(self, node: ast.Attribute) -> bool:
        """Validate attribute access."""
        attr_name = node.attr
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_ATTRS:
            if attr_name.startswith(pattern):
                return False
        return True


class SecureExecutor:
    """Executes code in a secure sandboxed environment."""

    def __init__(self, timeout: int = 30):
        """Initialize secure executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.validator = SecurityValidator()
        self._setup_builtins()

    def _setup_builtins(self) -> None:
        """Set up safe builtins for execution."""
        # Safe mathematical functions
        safe_math = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "pow": pow,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

        self.safe_builtins = safe_math

    def create_sandbox(self) -> dict[str, Any]:
        """Create a sandboxed execution environment.

        Returns:
            Dictionary of safe globals for code execution
        """
        import math

        sandbox = {
            "__builtins__": self.safe_builtins,
            "cq": cq,
            "cadquery": cq,
            "np": np,
            "numpy": np,
            "math": math,
            # Common CadQuery shortcuts
            "Workplane": cq.Workplane,
            "Vector": cq.Vector,
            "Location": cq.Location,
            "Assembly": cq.Assembly,
        }

        return sandbox

    def validate_code(self, code: str) -> None:
        """Validate code for security issues.

        Args:
            code: Python code to validate

        Raises:
            CodeValidationError: If code contains security issues
        """
        is_valid, error_msg = self.validator.validate(code)
        if not is_valid:
            raise CodeValidationError(code, error_msg or "Unknown validation error")

    @contextmanager
    def capture_output(self):
        """Context manager to capture stdout/stderr during execution."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield stdout_capture, stderr_capture
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute(
        self,
        code: str,
        sandbox: Optional[dict[str, Any]] = None,
        result_var: str = "result",
    ) -> Any:
        """Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            sandbox: Optional custom sandbox (uses default if None)
            result_var: Variable name to extract as result

        Returns:
            Value of result_var after execution, or None if not found

        Raises:
            CodeValidationError: If code validation fails
            CodeExecutionError: If code execution fails
            TimeoutError: If execution times out
        """
        # Validate code first
        self.validate_code(code)

        # Create sandbox
        if sandbox is None:
            sandbox = self.create_sandbox()

        # Set up timeout handler
        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("Code execution", self.timeout)

        # Execute with timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            with self.capture_output() as (stdout, stderr):
                exec(code, sandbox)

            # Extract result
            result = sandbox.get(result_var)

            # If no explicit result, try to find a Workplane
            if result is None:
                for value in sandbox.values():
                    if isinstance(value, cq.Workplane):
                        result = value
                        break

            return result

        except TimeoutError:
            raise
        except Exception as e:
            # Get captured output for debugging
            stdout_content = stdout.getvalue() if "stdout" in locals() else ""
            stderr_content = stderr.getvalue() if "stderr" in locals() else ""

            error_msg = str(e)
            if stdout_content:
                error_msg += f"\nStdout: {stdout_content}"
            if stderr_content:
                error_msg += f"\nStderr: {stderr_content}"

            raise CodeExecutionError(code, error_msg)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def execute_with_timeout(
        self,
        func: Callable[[], T],
        timeout: Optional[int] = None,
    ) -> T:
        """Execute a function with timeout.

        Args:
            func: Function to execute
            timeout: Timeout in seconds (uses instance timeout if None)

        Returns:
            Function result

        Raises:
            TimeoutError: If execution times out
        """
        timeout = timeout or self.timeout

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("Function execution", timeout)

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            return func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def execute_cad_code(
    code: str,
    timeout: int = 30,
    result_var: str = "result",
) -> cq.Workplane:
    """Execute CAD code and return the resulting Workplane.

    Args:
        code: CadQuery Python code
        timeout: Maximum execution time in seconds
        result_var: Variable name containing the result

    Returns:
        CadQuery Workplane object

    Raises:
        CodeValidationError: If code validation fails
        CodeExecutionError: If code execution fails
        TimeoutError: If execution times out
    """
    executor = SecureExecutor(timeout=timeout)
    result = executor.execute(code, result_var=result_var)

    if not isinstance(result, cq.Workplane):
        raise CodeExecutionError(
            code,
            f"Expected CadQuery Workplane, got {type(result).__name__}",
        )

    return result