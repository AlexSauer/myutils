import yaml
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple, Optional, Deque, Callable
from datetime import datetime
from collections import deque
import itertools
import gputils as gpu


@dataclass
class ExperimentParameter:
    base: dict  # Yaml file as dict with the base parameters
    extension: dict  # Yaml file as dict with the parameters to change for this experiment

    def to_str_tuple(self) -> Tuple[str, str]:
        return (yaml.safe_dump(self.base), yaml.safe_dump(self.extension))

    def update_device_id(self, new_id: int) -> None:
        if 'device_id' in self.base:
            self.base['device_id'] = new_id
        if 'DEVICE_ID' in self.base:
            self.base['DEVICE_ID'] = new_id


class HyperparameterSearch:
    """
    Takes a base yaml file looking like:
    para1: v1
    para2: v2
    ...
    paraK: v1
    ...
    paraN: vN

    and a second yaml file (extension) looking like
    paraK: [v1, v2, v3, ..., vM]
    ...
    paraJ: [v1, v2, v3, ..., VL]
    
    and will then go through the values provided in the extension
    file and run main with these parameters. (No grid search at the moment,
    i.e in the example above there would be M+L runs instead of M*L)
    One parameter has to be device_id which specifies an integer
    that will be used to select a GPU.
    """
    def __init__(self, base_file: str, extension_file: Optional[str] = None) -> None:
        with open(base_file, 'r') as f:
                self.base_dict = yaml.safe_load(f)

        if extension_file is not None:
            with open(extension_file, 'r') as f:
                self.extension_dict = yaml.safe_load(f)

        assert 'device_id' in set(self.base_dict.keys()).union(self.extension_dict.keys()) \
               or 'DEVICE_ID' in set(self.base_dict.keys()).union(self.extension_dict.keys()), \
               f'yaml files need to specify device_id integer for GPU.'

    def build_experiment_deque(self, grid_search: bool = False) -> Deque[ExperimentParameter]:
        """
        Build a deque of ExperimentParameter instances which either just 
        search for each parameter space individually or explore the interactions
        when grid_serach is set to True. 
        """
        experiment_parameter_deque = deque()
        if grid_search:
            # Build the grid to search through
            parameters, values = zip(*self.extension_dict.items())
            grid = [dict(zip(parameters, v)) for v in itertools.product(*values) ]
            for experiment in grid:            
                run_name = self.base_dict['run_name'] + ''.join([f'_{p}_{v}' for p, v in experiment.items()])
                experiment.update({'run_name':run_name})
                experiment_parameter_deque.append(ExperimentParameter(base=self.base_dict,
                                                                      extension=experiment))
        else:
            # If no grid search, go through all hyperparameters and combine the base with the specified values
            for parameter, possible_values in self.extension_dict.items():
                for cur_value in possible_values:
                    cur_para = {'run_name': f"{self.base_dict['run_name']}_{str(parameter)}{str(cur_value)}",
                                parameter: cur_value}
                    experiment_parameter_deque.append(ExperimentParameter(base=self.base_dict,
                                                                          extension=cur_para))
        return experiment_parameter_deque

    @staticmethod
    def run(func: Callable, experiment_parameter_deque: Deque[ExperimentParameter], timeout: int = 30,
            grid_serach: bool = False, max_processes: int = 4) -> None:
        node = gpu.Node()
        counter = 1
        processes = []
        gpus_allocated = []
        while experiment_parameter_deque:
            idle_gpus = node.find_idle_GPU(gpus_allocated)
            if idle_gpus and len(processes) < max_processes:
                cur_parameters = experiment_parameter_deque.popleft()
                cur_parameters.update_device_id(idle_gpus[0])
                print(f'Start process {counter} on GPU {idle_gpus[0]} '
                      f'at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                new_process = mp.Process(target=func, args=cur_parameters.to_str_tuple(),
                                            name=f'Process{counter}GPU{idle_gpus[0]}')
                new_process.start()
                processes.append(new_process)
                gpus_allocated.append(idle_gpus[0])
                counter += 1
                if max_processes == 1:
                    new_process.join()

            for p in processes:
                if not p.is_alive():
                    print(f'Process {p.name} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')
                    used_gpu = int(p.name[-1])
                    gpus_allocated.remove(used_gpu)
                    processes.remove(p)

            # Processes need some time to load in data and start using the gpu
            time.sleep(timeout)

        for p in processes:
            p.join()
