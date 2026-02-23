"""
Domain-specific visualization tools for climate risk disclosure analysis.

- DisclosureVisualizer: Climate risk disclosure patterns
- LocationVisualizer: Firm property and location data
- SimilarityVisualizer: Disclosure similarity metrics
- FinancialVisualizer: Financial data distributions

Generic panel visualizations (treatment summary, coverage, event study plots)
are in the ``did-panel-builder`` package.
"""

from .location_visualizer import LocationVisualizer
from .financial_visualizer import FinancialVisualizer

__all__ = [
    "LocationVisualizer",
    "FinancialVisualizer",
]
