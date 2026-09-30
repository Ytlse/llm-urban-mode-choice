from text_helper.templates.repository import tpl_describe_the_ob_wait_in_stop
from text_helper.type import EnvOb

class EnvObWaitInStop(EnvOb):
    duration: float = None
    stop_name: str = None
    by_vehicle_route_id: str = None

    def describe(self, weather=None) -> str:
        return tpl_describe_the_ob_wait_in_stop.render(ob=self, weather=weather)
    

if __name__ == "__main__":
    # Example usage
    event = EnvObWaitInStop(
        timestamp=1234567890,
        moving_id="123",
        duration=30.0,
        stop_name="Main St",
        by_vehicle_route_id="route_1"
    )
    print(event.describe())
