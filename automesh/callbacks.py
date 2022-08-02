import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from optuna import Trial

from automesh.models.heatmap import HeatMapRegressor

class OptimalMetric(Callback):
    def __init__(self, direction: str, monitor: str):
        super().__init__()
        assert direction == 'maximize' or direction == 'minimize'

        self.direction = direction
        self.monitor = monitor
        self.name = self.direction + '_' + self.monitor

    def on_validation_end(self, trainer: Trainer, _: HeatMapRegressor):
        if not trainer.is_global_zero:
            return

        ## grab current value from rank 0 trainer
        current_value = trainer.callback_metrics.get(self.monitor).item()

        ## best value has not been set yet
        if self.name not in trainer.callback_metrics:
            trainer.callback_metrics[self.name] = current_value
        else:

            ## get best value
            best_value = trainer.callback_metrics[self.name]
            maximum_optimal = self.direction == 'maximize' and current_value > best_value
            minimum_optimal = self.direction == 'minimize' and current_value < best_value

            ## update previous best value if better
            if maximum_optimal or minimum_optimal:
                trainer.callback_metrics[self.name] = current_value

class AutoMeshPruning(Callback):
    def __init__(self, trial: Trial, metric: str):
        super().__init__()

        self.trial = trial
        self.metric = metric

    def on_validation_end(self, trainer: Trainer, pl_module: HeatMapRegressor):
        score = trainer.callback_metrics.get(self.metric)

        assert isinstance(score, torch.Tensor)
        epoch = pl_module.current_epoch

        should_stop = False
        if trainer.is_global_zero:
            self.trial.report(score, epoch)
            should_stop = self.trial.should_prune()
            trainer.callback_metrics['pruned'] = should_stop

        trainer.should_stop = trainer.training_type_plugin.broadcast(should_stop)
        