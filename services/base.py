"""
services/__init__.py
--------------------
Service registry — maps .env names to service classes.

To add a new service:
  1. Create services/your_service.py
  2. Import it here and add to REGISTRY
  3. Add its name to PIPELINE in .env
"""

from .detector import DetectorService

# from .counter import CounterService
# from .tracker import TrackerService

REGISTRY = {
    "detector": DetectorService,
    # "counter" : CounterService,
    # "tracker" : TrackerService,
}