from pathlib import Path
import os
import yaml
import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import wandb

import carla

from oatomobile.core.rl import Env
from oatomobile.myscripts.myagents.baseAgent import BaseAgent
from oatomobile.myscripts.myagents.coilAgent.network.models.coil_icra import CoILICRA
from oatomobile.myscripts.myagents.coilAgent.network import Loss


def extract_modality(batch, modalities):
    return torch.cat([batch[k] for k in modalities], dim=-1)


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

    def log_example(self, batch, prefix, n=5):
        self._model.eval()
        with torch.no_grad():
            directions = batch['directions']
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            self._model.zero_grad()
            branches = self._model(torch.squeeze(batch['rgb'].to(self._device)),
                                   extract_modality(batch, self._inputs).to(self._device))
            loss_function_params = {
                'branches': branches,
                'targets': extract_modality(batch, self._targets).to(self._device),
                'directions': directions.to(self._device),
                'inputs': extract_modality(batch, self._inputs).to(self._device),
                'branch_weights': self.params["BRANCH_LOSS_WEIGHT"],
                'variable_weights': self.params["VARIABLE_WEIGHT"],
                'number_of_branches': self.params['MODEL_CONFIGURATION']['branches']["number_of_branches"]
            }
            loss, _ = self._criterion(loss_function_params)

        # Log a random position
        idx = np.arange(n)
        # TODO: does not include the speed branch
        output = self._model.extract_branch(torch.stack(branches[0:self.params['MODEL_CONFIGURATION']['branches']["number_of_branches"]]), directions)
        error = torch.abs(output - extract_modality(batch, self._targets).cuda())

        logs = {f'{prefix}direction': directions[idx].cpu().numpy(),
                f'{prefix}inputs': extract_modality(batch, self._inputs)[idx].cpu().numpy(),
                f'{prefix}predictions': output[idx].cpu().numpy(),
                f'{prefix}groundTruth': extract_modality(batch, self._targets)[idx].cpu().numpy(),
                f'{prefix}error': error[idx].cpu().numpy().tolist(),}
        imgs = []
        for i in idx:
            img = plt.imshow(np.transpose(torch.squeeze(batch['rgb'][i]).numpy(), [1, 2, 0]))
            plt.title(f"direction: {logs[f'{prefix}direction'][i][0]}\n"
                      f"gt: {np.array2string(logs[f'{prefix}groundTruth'][i], precision=3, floatmode='fixed')}\n"
                      f"pred: {np.array2string(logs[f'{prefix}predictions'][i], precision=3, floatmode='fixed', suppress_small=True)}")
            imgs.append(wandb.Image(img))
            plt.close()
        logs[f'{prefix}img'] = imgs
        return logs

    def eval_step(self, batch):
        self._model.eval()
        with torch.no_grad():
            directions = batch['directions']
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            self._model.zero_grad()
            branches = self._model(torch.squeeze(batch['rgb'].to(self._device)),
                                   extract_modality(batch, self._inputs).to(self._device))
            loss_function_params = {
                'branches': branches,
                'targets': extract_modality(batch, self._targets).to(self._device),
                'directions': directions.to(self._device),
                'inputs': extract_modality(batch, self._inputs).to(self._device),
                'branch_weights': self.params["BRANCH_LOSS_WEIGHT"],
                'variable_weights': self.params["VARIABLE_WEIGHT"],
                'number_of_branches': self.params['MODEL_CONFIGURATION']['branches']["number_of_branches"]
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

        directions = batch['directions']
        # directions = batch['mode']

        # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
        self._model.zero_grad()
        branches = self._model(torch.squeeze(batch['rgb'].to(self._device)),
                               extract_modality(batch, self._inputs).to(self._device))
        loss_function_params = {
            'branches': branches,
            'targets': extract_modality(batch, self._targets).to(self._device),
            'directions': directions.to(self._device),
            'inputs': extract_modality(batch, self._inputs).to(self._device),
            'branch_weights': self.params["BRANCH_LOSS_WEIGHT"],
            'variable_weights': self.params["VARIABLE_WEIGHT"],
            'number_of_branches': self.params['MODEL_CONFIGURATION']['branches']["number_of_branches"]
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

        # TODO: used for their weird lr schedule
        # loss_window.append(loss.data.tolist())

        return metrics