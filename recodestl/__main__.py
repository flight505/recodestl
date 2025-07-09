"""RecodeSTL - Convert STL files to parametric CAD models using AI."""

import sys
from typing import Optional

from recodestl.cli.app import app


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the RecodeSTL CLI."""
    try:
        app(argv)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())