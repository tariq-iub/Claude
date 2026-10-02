#!/usr/bin/env python3
"""Write header-only CSV templates for every results table (no values) into tables/."""
import _common  # noqa
from vpc.experiments.tables import write_templates
print("\n".join(write_templates("tables")))
