"""
NavLLM: LLM-Assisted Conversational Navigation over Multidimensional Data Cubes

A system that leverages LLMs as preference models and orchestration layers
on top of conventional cube engines for guided OLAP exploration.
"""

__version__ = "0.1.0"
__author__ = "NavLLM Team"

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level
from navllm.cube.view import CubeView
from navllm.nav.session import Session
from navllm.nav.recommender import NavLLMRecommender

__all__ = [
    "CubeSchema",
    "Dimension", 
    "Measure",
    "Level",
    "CubeView",
    "Session",
    "NavLLMRecommender",
]
