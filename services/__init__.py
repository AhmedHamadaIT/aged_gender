"""
services/__init__.py
--------------------
Service registry — maps .env PIPELINE names to service classes.

To add a new service:
    1. Create services/your_service.py
    2. Import it here and add to REGISTRY
    3. Add its name to PIPELINE in .env
    4. Restart container
"""

from .detector   import DetectorService
from .age_gender import AgeGenderService
from .ppe import PPEService
from .mood       import MoodService
from .cashier    import CashierService

# from .counter import CounterService
# from .tracker import TrackerService

REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
    "ppe"       :PPEService,
    "mood"      : MoodService,
    "cashier"   : CashierService,
    # "counter" : CounterService,
    # "tracker" : TrackerService,
}