from myutils.gputils.Process import Process
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import subprocess
import socket

class Node:
    def __init__(self, node: Optional[str] = None) -> None:
        if node is not None:
            self.node = node
        else:
            hostname = socket.gethostname()
            self.node = hostname[:hostname.find('.')]
    
    def query(self) -> Dict[int, List[Process]]:
        """
        Runs nvidia-smi on the node and processes the output
        to give which gpus run which processes at the moment.
        An empty list means the GPU is idle
        """
        command = "nvidia-smi -x -q"
        result = subprocess.run([command], stdout=subprocess.PIPE, shell = True)
        result = result.stdout.decode('utf-8')

        # Parse the xml string and find the associated processes
        root = ET.fromstring(result)
        return self.process_XML(root)
    
    def process_XML(self, root: ET, verbose = False) -> Dict[int, List[Process]]:
        """
        Responsible for the parsing of the xml object returned by 
        nvidia-smi which is given as <root>. 
        It prints out which processes are currently running on 
        which GPU if verbose and returns a dict with all 
        the processes found on this node of the form:
        {gpu1: [Process1, ..., ProcessK],
         ...,
         gpuN: [Process1, ..., ProcessJ]}
        """
        processes_on_node = {}
        for gpu_id, gpu in enumerate(root.findall('gpu')):
            processes_on_gpu = []
            for p in gpu.findall('processes'):           
                # Check if there are no running processes
                if len(p) == 0:
                    continue
                # If we have running proccess, then:
                for spec_proc in p.findall('process_info'):
                    process = Process(self.node,
                                    gpu_id,  
                                    spec_proc.find('pid').text,
                                    spec_proc.find('type').text,
                                    spec_proc.find('process_name').text,
                                    spec_proc.find('used_memory').text)
                    processes_on_gpu.append(process)

            # Print the processes found:
            if verbose:
                if len(processes_on_gpu) == 0:
                    print('GPU: ', gpu_id, ' No running processes!')
                for p in processes_on_gpu:
                    print('GPU: ', gpu_id,  p.to_string())

            processes_on_node[gpu_id] = processes_on_gpu
        
        return processes_on_node

    def find_available_GPUs(self) -> List[int]:
        """Returns a list of available GPUs"""
        GPU_processes = self.query()
        return list(GPU_processes.keys())

    def find_idle_GPU(self, reserved_gpus: List[int]) -> List[int]:
        """Returns a list of idle GPUs and respected gpus which are reserved"""
        GPU_processes = self.query()
        idle_GPUs = [gpu for gpu, processes in GPU_processes.items() if not processes]
        idle_GPUs = [gpu for gpu in idle_GPUs if gpu not in reserved_gpus]
        return idle_GPUs

    def __repr__(self) -> str:
        return self.node

    
