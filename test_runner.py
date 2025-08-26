#!/usr/bin/env python3
"""
Test runner script for the transformer project.

This script provides an easy way to run the transformer tests with enhanced output.
"""

import sys
import subprocess
import time
from pathlib import Path
import traceback


def print_banner(title, char="=", width=80):
    """Print a formatted banner."""
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width)


def print_section(title, char="-", width=60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"🔧 {title}")
    print(f"{char * width}")


def run_unittest():
    """Run tests using the built-in unittest framework."""
    print_section("Running Tests with unittest")
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_transformer", "-v"],
        cwd=Path(__file__).parent,
        check=False,
    )

    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ All tests passed in {duration:.2f}s!")
    else:
        print(f"\n❌ Tests failed after {duration:.2f}s")

    return result.returncode


def run_pytest():
    """Run tests using pytest with beautiful output."""
    print_section("Running Tests with pytest + Rich Output")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--color=yes",
                "--durations=10",
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed in {duration:.2f}s!")
            print("📊 HTML report generated: reports/report.html")
        else:
            print(f"\n❌ Tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest not found. Installing...")
        # Try to install pytest
        install_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-html",
                "pytest-cov",
                "pytest-sugar",
                "rich",
            ],
            check=False,
        )
        if install_result.returncode == 0:
            print("✅ pytest installed successfully! Running tests...")
            return run_pytest()
        print("❌ Failed to install pytest. Falling back to unittest...")
        return run_unittest()


def run_pytest_with_coverage():
    """Run tests using pytest with coverage report and beautiful output."""
    print_section("Running Tests with Coverage Analysis + Rich Output")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:reports/coverage",
                "-v",
                "--tb=short",
                "--color=yes",
                "--durations=10",
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed with coverage in {duration:.2f}s!")
            print("📊 HTML report: reports/report.html")
            print("📈 Coverage report: reports/coverage/index.html")
        else:
            print(f"\n❌ Tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest-cov not found. Installing...")
        install_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pytest",
                "pytest-cov",
                "pytest-html",
                "pytest-sugar",
                "rich",
            ],
            check=False,
        )
        if install_result.returncode == 0:
            print("✅ pytest installed successfully! Running tests...")
            return run_pytest_with_coverage()
        print("❌ Failed to install pytest. Falling back to unittest...")
        return run_unittest()


def run_pytest_parallel():
    """Run tests in parallel using pytest-xdist."""
    print_section("Running Tests in Parallel")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "-n",
                "auto",  # Auto-detect number of CPUs
                "--html=reports/report.html",
                "--self-contained-html",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ All tests passed in parallel in {duration:.2f}s!")
        else:
            print(f"\n❌ Parallel tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print("❌ pytest-xdist not found. Install with: pip install pytest-xdist")
        return run_pytest()


def run_benchmark_tests():
    """Run benchmark tests only."""
    print_section("Running Benchmark Tests")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--benchmark-only",
                "--benchmark-sort=mean",
                "-v",
                "--tb=short",
                "--color=yes",
            ],
            cwd=Path(__file__).parent,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ Benchmark tests completed in {duration:.2f}s!")
        else:
            print(f"\n❌ Benchmark tests failed after {duration:.2f}s")

        return result.returncode

    except FileNotFoundError:
        print(
            "❌ pytest-benchmark not found. Install with: pip install pytest-benchmark"
        )
        return run_pytest()


def run_standalone_test():
    """Run a simple standalone test to check basic functionality."""
    print_section("Running Standalone Test")
    start_time = time.time()

    try:
        # Add the current directory to Python path
        sys.path.insert(0, str(Path(__file__).parent))
        from tests.test_transformer import standalone_test_run # pylint: disable=import-outside-toplevel

        config_file = Path("config.toml")
        if not config_file.exists():
            print(f"❌ Configuration file {config_file} not found!")
            return 1

        print("🚀 Starting standalone test...")
        standalone_test_run(config_file)

        duration = time.time() - start_time
        print(f"\n✅ Standalone test completed successfully in {duration:.2f}s!")
        return 0

    except ImportError as e:
        duration = time.time() - start_time
        print(f"\n❌ Import error after {duration:.2f}s: {e}")
        print("💡 Make sure the tests directory contains __init__.py")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        duration = time.time() - start_time
        print(f"\n❌ Error running standalone test after {duration:.2f}s: {e}")
        traceback.print_exc()
        return 1


def show_help():
    """Show help message."""
