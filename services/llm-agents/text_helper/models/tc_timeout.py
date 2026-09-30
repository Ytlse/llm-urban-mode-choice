from text_helper.type import EnvOb


class EnvObTcTimeout(EnvOb):
    wait_duration: float
    route_id: str
    stop_name: str

    def describe(self) -> str:
        minutes = int(self.wait_duration / 60)
        return f"I missed my bus on route {self.route_id} at stop {self.stop_name} after waiting {minutes} minutes."
