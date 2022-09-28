import yaml
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple, Optional, Deque, Callable
from datetime import datetime
from collections import deque
from gputils import gpu

@dataclass
class ExperimentParameter:
    base: str  # Yaml file as string with the base parameters
    extension: str  # Yaml file as string with the parameters to change for this experiment

    def to_str_tuple(self) -> Tuple[str, str]:
        return (yaml.safe_dump(self.base), yaml.safe_dump(self.extension))

    def update_device_id(self, new_id: int) -> None:
        if 'device_id' in self.base:
            self.base['device_id'] = new_id
        if 'DEVICE_ID' in self.base:
            self.base['DEVICE_ID'] = new_id


class HyperparameterSearch:
    """
    Takes a base file looking like:
    para1: v1
    para2: v2
    ...
    paraN: vN

    and a second yaml file looking like
    paraK: [v1, v2, v3, ..., vM]
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

    def build_experiment_deque(self) -> Deque[ExperimentParameter]:
        experiment_parameter_deque = deque()
        for parameter, possible_values in self.extension_dict.items():
            for cur_value in possible_values:
                cur_para = {'run_name': f"{self.base_dict['run_name']}_{str(parameter)}{str(cur_value)}",
                             parameter: cur_value}
                experiment_parameter_deque.append(ExperimentParameter(base=self.base_dict,
                                                                      extension=cur_para))
        return experiment_parameter_deque

    @staticmethod
    def run(func: Callable, experiment_parameter_deque: Deque[ExperimentParameter], timeout: int = 30,
            max_processes: int = 4) -> None:
        node = gpu.Node()
        counter = 0
        processes = []
        gpus_allocated = []
        while experiment_parameter_deque:
            idle_gpus = node.find_idle_GPU(gpus_allocated)
            if idle_gpus and len(processes) < max_processes:
                cur_parameters = experiment_parameter_deque.popleft()
                cur_parameters.update_device_id(idle_gpus[0])
                print(f'Start process {counter} on GPU {idle_gpus[0]} '
                      f'at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                if max_processes > 1:
                    new_process = mp.Process(target=func, args=cur_parameters.to_str_tuple(),
                                             name=f'Process{counter}GPU{idle_gpus[0]}')
                    new_process.start()
                    processes.append(new_process)
                    gpus_allocated.append(idle_gpus[0])
                    counter += 1
                else:
                    func(cur_parameters.to_str_tuple())
                    print(f'Process {p.name} finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}!')

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
