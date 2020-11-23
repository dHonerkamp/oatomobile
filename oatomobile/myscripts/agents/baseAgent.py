from typing import Any
import numpy as np
import torch

import carla
import oatomobile
from oatomobile.core.rl import Env
from oatomobile.core.simulator import Action, Observations


class BaseAgent(oatomobile.Agent):
    def __init__(self, environment: Env, model, checkpoint, *args: Any, **kwargs: Any) -> None:
        super(BaseAgent, self).__init__(environment)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = model
        # Load the model and prepare set it for evaluation
        if checkpoint:
            self._model.load_state_dict(checkpoint['state_dict'])
        self._model.to(self._device)
        self._model.eval()

    def act(self, observation: oatomobile.Observations, *args,
            **kwargs) -> oatomobile.Action:

        # TODO: adapt signature or parse obs
        return self.run_step()

    def run_step(self, measurements, sensor_data, directions, target):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: All the float measurements from CARLA ( Just speed is used)
            sensor_data: All the sensor data used on this benchmark
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """
        # TODO: check the stuff coilAgent does and maybe include for all of them
        raise NotImplementedError()

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control


    def update(
            self,
            observations: Observations,
            action: Action,
            new_observations: Observations,
    ) -> None:
        """Updates the agent given a transition."""
        del observations
        del action
        del new_observations

    def train_step(self, batch) -> dict:
        raise NotImplementedError("Nope")

    def eval_step(self, batch):
        raise NotImplementedError("Nope")
