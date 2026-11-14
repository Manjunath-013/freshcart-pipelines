# conftest.py — adds the project root to sys.path so pytest can import iris_project
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

