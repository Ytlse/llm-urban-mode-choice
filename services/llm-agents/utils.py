import asyncio
from typing import Coroutine

from faker import Faker
from pyproj import Transformer
from models import Location
from settings import settings

fake = Faker("fr_FR")

# Strong references on fire-and-forget tasks: the event loop only keeps
# weakrefs on tasks, an unreferenced task can be garbage-collected before
# (or during) its execution. The set keeps the reference until completion.
_background_tasks: set = set()


def create_background_task(coro: Coroutine) -> "asyncio.Task":
    """asyncio.create_task keeping a reference until the task finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

def random_name():
    return fake.name()

def random_uuid():
    return fake.uuid4()

def random_choices(elements, length):
    return fake.random_choices(
        elements=elements,
        length=length,
    )

transformer = Transformer.from_crs(settings.world.geo_crs, settings.world.geo_projection) 
def world_projection(location: Location) -> tuple[float, float]:
    """
    Project a point from WGS84 to Web Mercator
    """
    return transformer.transform(location.lon, location.lat)

def square_distance(p1: Location, p2: Location) -> float:
    """
    Calculate the square distance between two points
    """
    x1, y1 = world_projection(p1)
    x2, y2 = world_projection(p2)
    return (x1 - x2) ** 2 + (y1 - y2) ** 2    
