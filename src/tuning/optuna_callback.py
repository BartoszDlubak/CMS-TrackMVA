import optuna


class OptunaCallbackTorch:
    def __init__(self, trial, monitor="valid_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, train_loss, valid_loss):
        self.trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)
        self.trial.set_user_attr(f"valid_loss_epoch_{epoch}", valid_loss)
        
        metric = valid_loss if self.monitor == "valid_loss" else train_loss
        self.trial.report(metric, step=epoch)

        # check if pruning should occur
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")