from dataclasses import dataclass

@dataclass
class Process:
    """ 
    Simple class the save all the relevant information we have about a GPU process
    """
    node: str
    gpu: str
    pid: str
    type: str
    process_name: str
    memory: str

    def to_string(self) -> str:
        return '{:>8}  {:>32}  {:>14}'.format(self.pid, self.process_name[-32:], self.memory)

    def __repr__(self) -> str:
        return '(Node: {}, GPU: {}, PID: {}, Type: {}, Name: {}, Memory: {})'.format(
                self.node, self.gpu, self.pid, self.type, self.process_name, self.memory)

