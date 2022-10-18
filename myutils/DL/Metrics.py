import torch
import numpy as np
from typing import Union, List

class Metrics:
    def __init__(self, metrics: str = '') -> None:
        requested_metrics = metrics.lower().split(' ')
        self.metrics = {}
        self.lookup = {'Io'}
        for metric in requested_metrics:
            assert f'comp_{metric}_bool' in dir(self), \
                   f'Method to compute {metric} is not implemented yet!'
            self.metrics[metric] = []


    def comp_metrics_from_bool(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Takes two boolean tensors and computes the metrics in self.metrics for them 
        """
        for metric in self.metrics.keys():
            self.metrics[metric].append(getattr(self, f'comp_{metric}_bool')(pred, target))

    def get(self, metrics: str) -> Union[float, List[float]]:
        """
        Splits the metrics string by white spaces and returns the current value 
        If there is only a single metric requested, it just returns that float 
        """
        res = []
        for metric in metrics.lower().split(' '):
            res.append(self.metrics[metric])
        if len(res) == 1:
            return res[0]
        else:
            return res

    def get_last(self, metrics: str) -> Union[float, List[float]]:
        """
        Splits the metrics string by white spaces and returns the most recent value computed
        If there is only a single metric requested, it just returns that float 
        """
        res = []
        for metric in metrics.lower().split(' '):
            res.append(self.metrics[metric][-1])
        if len(res) == 1:
            return res[0]
        else:
            return res
        
    def summarise_metrics(self) -> None:
        """
        Replace the list of computed metrics with its mean 
        """
        for metric in self.metrics.keys():
            self.metrics[metric] = np.mean(self.metrics[metric])

    def add_metric_tensorboard(self, writer: "SummaryWriter", iteration: int) -> None: 
        """
        Add all the metrics to the summary writer. 
        """
        for metric in self.metrics.keys():
            label = self._transform_metric_label(metric)
            assert isinstance(self.metrics[metric], float), f'{metric} is not a float!'
            writer.add_scalar(label, self.metrics[metric], iteration)


    def _transform_metric_label(self, label: str) -> str:
        if label == 'iou':
            return 'IoU'
        else:
            return label.capitalize()


    @staticmethod
    def _check_and_flatten(pred, target):
        assert pred.shape == target.shape, "Shapes don't match!"
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        return pred, target

    @staticmethod
    def comp_iou_bool(pred, target):
        # Compute the IoU per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        inter = torch.sum(torch.logical_and(pred, target), dim=1)
        union = torch.sum(torch.logical_or(pred, target), dim=1)
        iou = inter / (union + 1e-8)
        return torch.mean(iou).item()

    @staticmethod
    def comp_dice_bool(pred, target):
        # Compute the dice per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)

        inter = torch.sum(torch.logical_and(pred, target), dim=1)
        denom = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
        dice = 2 * inter / (denom + 1e-8)
        return torch.mean(dice).item()

    @staticmethod
    def comp_accuracy_bool(pred, target):
        # Compute the dice per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        overlap = torch.sum(pred == target, dim=1)
        accuracy = overlap / target.shape[1]
        return torch.mean(accuracy).item()

    @staticmethod
    def comp_precision_bool(pred, target):
        # Compute the IoU per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        true_positive = torch.logical_and(pred, target).sum(dim=1)
        pred_positive = pred.sum(dim=1)
        precision = true_positive / pred_positive
        return torch.mean(precision).item()

    @staticmethod
    def comp_recall_bool(pred, target):
        # Compute the IoU per image and only take the mean at the end
        pred, target = Metrics._check_and_flatten(pred, target)
        true_positive = torch.logical_and(pred, target).sum(dim=1)
        positive = target.sum(dim=1)
        recall = true_positive / positive
        return torch.mean(recall).item()