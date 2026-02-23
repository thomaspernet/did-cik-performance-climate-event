"""
Treatment matching module.

Matches firm production locations to SHELDUS disaster events.
Treatment assignment (event indicators, filtering) is handled by
``did_panel_builder.TreatmentAssigner``.
"""

from .sheldus_treatment_builder import SheldusTreatmentBuilder

__all__ = ["SheldusTreatmentBuilder"]
