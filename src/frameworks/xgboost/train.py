
import xgboost as xgb
import numpy as np

class XGBTrainer():
    def __init__(self, params, num_boost_round, early_stopping_rounds, L_FP, L_FN, save_path):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.save_path = save_path
        self.L_FP = L_FP
        self.L_FN = L_FN
        
        self.model = None
        
        self.loss_obj = weighted_logloss_obj
        self.loss_eval = weighted_logloss_eval

    def train(self, X_train, X_val, y_train, y_val, callbacks=None):
        w_train = compute_weights(y_train, self.L_FP, self.L_FN)
        w_val   = compute_weights(y_val, self.L_FP, self.L_FN)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval   = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        evals = [(dtrain, "train"), (dval, "valid")]
        
        evals_result = {}
        self.model = xgb.train(
            params = self.params,
            dtrain = dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            evals_result=evals_result,
            obj = self.loss_obj,
            custom_metric= self.loss_eval,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=True,
            # callbacks=[XGBoostPruningCallback(trial, "validation-logloss")]
            )

        self.model.save_model(self.save_path)

        # Callbacks (e.g. Optuna) if provided
        if callbacks:
            best_score = getattr(self.model, "best_score", None)
            for cb in callbacks:
                cb.on_epoch_end(self.model.best_iteration, best_score, None)

        return self.model

    def evaluate(self, X, y):
        dtest = xgb.DMatrix(X, label=y)
        preds = self.model.predict(dtest)
        return preds
    
    def get_loss(self, X, y):
        dtest = xgb.DMatrix(X, label=y)
        preds = self.model.predict(dtest)
        loss = self.loss_eval(preds, dtest)[1]
        return loss
    
    
# ----- Losses -----    

def compute_weights(y, L_FP, L_FN):
    y = np.asarray(y)
    w1 = y.mean(); w0 = 1.0 - w1
    w_pos = L_FN / w1
    w_neg = L_FP / w0
    c = 1.0 / (w_pos + w_neg)  
    return np.where(y == 1, w_pos * c, w_neg * c)


def weighted_logloss_obj(preds, dmatrix):
    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    p = sigmoid(preds)
    grad = (p - y)                       
    hess = p * (1.0 - p)                 
    # if w is not None and len(w) > 0:
    #     grad *= w; hess *= w             
    return grad, hess


def weighted_logloss_eval(preds, dmatrix):
    y = dmatrix.get_label()
    w = dmatrix.get_weight()
    if w is None or len(w) == 0:
        w = np.ones_like(y)
    p = 1.0 / (1.0 + np.exp(-preds))
    eps = 1e-12
    loss = - (w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / w.sum()
    return 'wlogloss', float(loss)





def relative_squared_error(preds, dmatrix):
    labels = dmatrix.get_label()
    loss = ((preds / labels) - 1) ** 2
    return 'relative_squared_error', float(np.mean(loss))



def relative_squared_error_obj(preds, dmatrix):
    labels = dmatrix.get_label()
    grad = 2 * (preds - labels) / (labels ** 2)
    hess = 2 / (labels ** 2)
    return grad, hess



### --- utils ----
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))