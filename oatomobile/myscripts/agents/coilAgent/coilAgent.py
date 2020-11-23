from pathlib import Path
import os
import yaml
import torch
import torch.optim as optim
import numpy as np

import carla

from oatomobile.core.rl import Env
from oatomobile.myscripts.agents.baseAgent import BaseAgent
from oatomobile.myscripts.agents.coilAgent.network.models.coil_icra import CoILICRA
from oatomobile.myscripts.agents.coilAgent.network import Loss


class CoilAgent(BaseAgent):
    def __init__(self, environment: Env, checkpoint) -> None:
        with open(Path(__file__).parent / "resnet34imnet10S1.yaml", 'r') as f:
            self.params = yaml.safe_load(f)

        model_params = self.params["MODEL_CONFIGURATION"]
        model = CoILICRA(model_params, self.params)

        # ['steer', 'throttle', 'brake']
        self._targets = self.params["TARGETS"]
        # -> 'velocity'?
        self._inputs = self.params["INPUTS"]
        self._criterion = Loss(self.params["LOSS_FUNCTION"])
        self._optimizer = optim.Adam(model.parameters(), lr=self.params["LEARNING_RATE"])

        super(CoilAgent, self).__init__(environment, model, checkpoint)

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

        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / self.params["SPEED_FACTOR"]
        norm_speed = torch.Tensor([norm_speed]).unsqueeze(0).to(self._device)
        directions_tensor = torch.Tensor([directions]).to(self._device)
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(self._process_sensors(sensor_data), norm_speed,
                                                  directions_tensor)

        steer, throttle, brake = self._process_model_outputs(model_outputs[0])

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        return control

    def _process_sensors(self, sensors):
        # TODO: update
        iteration = 0
        for name, size in self.params["SENSORS"].items():
            # sensor = sensors[name][self.params["IMAGE_CUT"][0]:self.params["IMAGE_CUT"][1], ...]
            # sensor = scipy.misc.imresize(sensor, (size[1], size[2]))

            sensor = np.swapaxes(sensor, 0, 1)
            sensor = np.transpose(sensor, (2, 1, 0))
            sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)
            iteration += 1

        image_input = image_input.unsqueeze(0)
        return image_input

    def _process_model_outputs(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:
        """
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        if brake < 0.05:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        return steer, throttle, brake

    def eval_step(self, batch):
        self._model.eval()
        with torch.no_grad():
            controls = batch['directions']
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            self._model.zero_grad()
            branches = self._model(torch.squeeze(batch['rgb'].to(self._device)),
                                   batch[self._inputs].to(self._device))
            loss_function_params = {
                'branches': branches,
                'targets': batch[self._targets],
                'controls': controls.to(self._device),
                'inputs': batch[self._inputs].to(self._device),
                'branch_weights': self.params["BRANCH_LOSS_WEIGHT"],
                'variable_weights': self.params["VARIABLE_WEIGHT"]
            }
            loss, _ = self._criterion(loss_function_params)

        return {'loss': loss.data}

    def train_step(self, batch) -> dict:
        self._model.train()

        # Basically in this mode of execution, we validate every X Steps, if it goes up 3 times,
        # add a stop on the _logs folder that is going to be read by this process
        # if g_conf.FINISH_ON_VALIDATION_STALE is not None and \
        #         check_loss_validation_stopped(iteration, g_conf.FINISH_ON_VALIDATION_STALE):
        #     break

        # TODO: put into main loop
        # iteration += 1
        # if iteration % 1000 == 0:
        #     adjust_learning_rate_auto(optimizer, loss_window)

        # get the control commands from float_data, size = [120,1]

        controls = batch['directions']
        # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
        self._model.zero_grad()
        branches = self._model(torch.squeeze(batch['rgb'].to(self._device)),
                               batch[self._inputs].to(self._device))
        loss_function_params = {
            'branches': branches,
            'targets': batch[self._targets],
            'controls': controls.to(self._device),
            'inputs': batch[self._inputs].to(self._device),
            'branch_weights': self.params["BRANCH_LOSS_WEIGHT"],
            'variable_weights': self.params["VARIABLE_WEIGHT"]
        }
        loss, _ = self._criterion(loss_function_params)
        loss.backward()
        self._optimizer.step()

        # TODO: checkpointing
        # if is_ready_to_save(iteration):
        #     state = {
        #         'iteration': iteration,
        #         'state_dict': self._model.state_dict(),
        #         'best_loss': best_loss,
        #         'total_time': accumulated_time,
        #         'optimizer': optimizer.state_dict(),
        #         'best_loss_iter': best_loss_iter
        #     }
        #     torch.save(state, os.path.join('_logs', exp_batch, exp_alias
        #                                    , 'checkpoints', str(iteration) + '.pth'))

        # logging
        metrics = {'loss': loss.data,
                   # 'image': torch.squeeze(batch['rgb']),
                   }


        # Log a random position
        # position = random.randint(0, len(data) - 1)
        #
        # output = self._model.extract_branch(torch.stack(branches[0:4]), controls)
        # error = torch.abs(output - dataset.extract_targets(data).cuda())
        #
        # accumulated_time += time.time() - capture_time
        #
        # coil_logger.add_message('Iterating',
        #                         {'Iteration': iteration,
        #                          'Loss': loss.data.tolist(),
        #                          'Images/s': (iteration * g_conf.BATCH_SIZE) / accumulated_time,
        #                          'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
        #                          'Output': output[position].data.tolist(),
        #                          'GroundTruth': dataset.extract_targets(data)[
        #                              position].data.tolist(),
        #                          'Error': error[position].data.tolist(),
        #                          'Inputs': dataset.extract_inputs(data)[
        #                              position].data.tolist()},
        #                         iteration)
        # coil_logger.write_on_error_csv('train', loss.data)
        # print("Iteration: %d  Loss: %f" % (iteration, loss.data))

        # TODO: used for their weird lr schedule
        # loss_window.append(loss.data.tolist())

        return metrics