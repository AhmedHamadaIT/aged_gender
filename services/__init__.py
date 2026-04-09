"""
services/__init__.py
--------------------
Service and task registries.

REGISTRY     — simple per-frame services used internally by FrameBus and tests.
TASK_REGISTRY — full task classes, each instantiated with a task config dict.
                Used by task_worker.py to spawn the right algorithm per task.

To add a new task:
    1. Create services/your_task.py with a class that accepts task_config in __init__
       and returns a list of event dicts from __call__(payload).
    2. Import it here and add to TASK_REGISTRY.
    3. Register the algorithmType string in apis/tasks.py → TaskRegistry.SUPPORTED.
"""

from .detector   import DetectorService
from .age_gender import AgeGenderService
from .ppe        import PPEService
from .mood       import MoodService
from .cross_line            import CrossLineTask
from .mask_hairnet_chef_hat import MaskHairnetChefHatTask

# Simple per-frame services (used by FrameBus internals and legacy code)
REGISTRY = {
    "detector"  : DetectorService,
    "age_gender": AgeGenderService,
    "ppe"       : PPEService,
    "mood"      : MoodService,
}

# Full task classes — keyed by algorithmType string from task config
TASK_REGISTRY = {
    "CROSS_LINE"          : CrossLineTask,
    "MASK_HAIRNET_CHEF_HAT": MaskHairnetChefHatTask,
}
