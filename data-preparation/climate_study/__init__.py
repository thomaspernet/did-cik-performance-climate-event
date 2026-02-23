"""
Climate Study Module — Domain-specific analysis for climate risk disclosure research.

This module provides tools specific to the climate risk detection project:
- SheldusTreatmentBuilder: Match firm locations to SHELDUS disaster events
- ClimateDisclosureBuilder: Build climate disclosure panels from Neo4j
- IntensiveMarginBuilder: Intensive-margin analysis variables
- DisclosureSimilarityMetrics: Compute disclosure similarity metrics

Generic DiD panel construction and visualization are in the
``did-panel-builder`` package.

Usage:
    from climate_study import SheldusTreatmentBuilder, ClimateDisclosureBuilder
    from climate_study.visualization import DisclosureVisualizer, LocationVisualizer
"""

from .treatment import SheldusTreatmentBuilder
#from .disclosure import IntensiveMarginBuilder
from .visualization import (
    LocationVisualizer,
    FinancialVisualizer,
)

__all__ = [
    # Treatment matching
    "SheldusTreatmentBuilder",
    # Disclosure builders
    #"IntensiveMarginBuilder",
    # Similarity metrics
    # Visualization
    "LocationVisualizer",
    "FinancialVisualizer",
]
