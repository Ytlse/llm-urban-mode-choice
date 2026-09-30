from typing import Callable, Coroutine, Optional
from urban_mobility_agents.agents.llm_agent import LlmAgent
from world.population import WorldPopulation
from urban_mobility_agents.core.schemas import Action, Observation

class BaseScenario:
    agent: "LlmAgent"

    async def sync(self, timestamp: int):
        """Synchronize the scenario with a given timestamp"""
        raise NotImplementedError("This method should be overridden by subclasses")

    async def handle_observation(self, observation: Observation):
        """Handle observation data"""
        raise NotImplementedError("This method should be overridden by subclasses")

    async def has_messages(self) -> bool:
        """Check if there are messages to process"""
        raise NotImplementedError("This method should be overridden by subclasses")

    async def pop_all_messages(self) -> list[Action]:
        """Pop all messages from the queue"""
        raise NotImplementedError("This method should be overridden by subclasses")

    @property
    def population(self) -> "WorldPopulation":
        """Get the current population of the scenario"""
        raise NotImplementedError("This method should be overridden by subclasses")

    def set_push_fn(self, fn: Callable[[Action], Coroutine]) -> None:
        """Inject the direct WebSocket push coroutine used by Worker and arrival handler."""
        raise NotImplementedError("This method should be overridden by subclasses")

    async def start_worker(self) -> None:
        """Start the background async Worker that consumes the activity queue."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def stop_worker(self) -> None:
        """Cancel the background Worker loop when the scenario is replaced."""
        raise NotImplementedError("This method should be overridden by subclasses")

    @property
    def worker_in_progress_count(self) -> int:
        """Return the number of activities currently queued or being computed by the Worker."""
        raise NotImplementedError("This method should be overridden by subclasses")

    @property
    def activities_to_compute_count(self) -> int:
        """Return activities in flight + idle agents without a plan at last sync.
        Used by /sync for backpressure (N in the min_interval formula)."""
        raise NotImplementedError("This method should be overridden by subclasses")
