import optuna


class OptunaCallbackTorch:
    def __init__(self, trial, monitor="valid_loss", patience=10):
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.best_loss = float("inf")
        self.no_improve_epochs = 0

    def on_epoch_end(self, epoch, train_loss, valid_loss):
        # record attributes
        self.trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)
        self.trial.set_user_attr(f"valid_loss_epoch_{epoch}", valid_loss)

        # choose metric
        metric = valid_loss if self.monitor == "valid_loss" else train_loss
        self.trial.report(metric, step=epoch)

        # check improvement
        if metric < self.best_loss:
            self.best_loss = metric
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

        # early stopping via patience
        if self.no_improve_epochs >= self.patience:
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} due to no improvement for {self.patience} epochs"
            )

        # pruner decision
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")
        