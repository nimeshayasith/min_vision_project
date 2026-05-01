"""
conftest.py — pytest configuration
Adds the project root to sys.path so that `from src.xxx import yyy` works.
"""
import sys
from pathlib import Path

# Insert project root at the beginning of sys.path
sys.path.insert(0, str(Path(__file__).parent))
