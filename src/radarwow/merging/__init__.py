"""radar-wow stage 7: gauge merging (Goudenhoofdt & Delobbe 2009 set of methods) and leave-one-out verification."""

from .methods import (METHODS, brandes, conditional_merging, krige, kriging_external_drift, leave_one_out, mean_field_bias, merge,
                      ordinary_kriging, radar_at)

PLANNED = ["RangeDependentAdjustment", "REALEnsemble"]
__all__ = ["METHODS", "brandes", "conditional_merging", "krige", "kriging_external_drift", "leave_one_out", "mean_field_bias", "merge",
           "ordinary_kriging", "radar_at"]
