"""Interface for ES RNN for Time Series Forecasting."""

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.es_rnn import ESRNN
from sktime.utils.dependencies import _check_soft_dependencies


class ESRNNForecaster(BaseDeepNetworkPyTorch):
    """
    Exponential Smoothing Recurrant Neural Network.

    Parameters
    ----------
    input_shape : int
        Number of features in the input

    hidden_size : int
        Number of features in the hidden state

    horizon : int
        Forecasting horizon

    num_layer : int
        Number of layers

    season_length : int
        Period of season

    seasonality : string
        Type of seasonality

    level_coeff : int

    seasonal_coeff_1 : int

    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ankit-1204"],
        "python_dependencies": ["torch"],
    }

    def __init__(
        self,
        input_shape=1,
        hidden_size=1,
        num_layer=1,
        season_length=3,
        seasonality="single",
        window=5,
        stride=1,
        batch_size=32,
        epoch=50,
        optimizer="Adam",
        criterion="pinball",
    ) -> None:
        super().__init__()
        if _check_soft_dependencies("torch", severity="none"):
            import torch
            import torch.nn as nn
        else:
            raise ImportError("torch is not available. Please install torch first.")

        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.seasonality = seasonality
        self.season_length = season_length
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr = 1e-5
        self.loss_list = {
            "mse": nn.MSELoss,
            "cross": nn.CrossEntropyLoss,
            "l1": nn.L1Loss,
        }
        self.opti_list = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    def _get_windows(self, y):
        length = len(y)
        x_arr = []
        y_arr = []
        for i in range(0, length - self.window - self.horizon + 1, self.stride):
            inp = y[i : i + self.window]
            out = y[i + self.window : i + self.window + self.horizon]

            x_arr.append(inp)
            y_arr.append(out)

        if not x_arr:
            raise ValueError("Input size to small")

        return np.array(x_arr), np.array(y_arr)

    def _instantiate_optimizer(self):
        import torch

        if self.optimizer:
            if self.optimizer.lower() in self.opti_list:
                return self.opti_list[self.optimizer.lower()](
                    self.network.parameters(),
                    lr=self.lr,
                )
            else:
                raise TypeError(
                    f"Please pass one of {self.opti_list.keys()} for `optimizer`."
                )
        else:
            return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _instantiate_criterion(self):
        if self.criterion:
            if self.criterion in self.loss_list:
                return self.loss_list[self.criterion]()
            else:
                loss = ESRNN().DefaultLoss()
                return loss
        else:
            # default criterion
            loss = ESRNN().DefaultLoss()
            return loss

    def _fit(self, y, fh, X=None):
        """Fit ES-RNN Model for provided data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._y = y.copy()
        self.horizon = len(fh)
        self._fh = fh
        self.input_shape = y.shape[1]
        self.network = ESRNN(
            self.input_shape,
            self.hidden_size,
            self.horizon,
            self.num_layer,
            self.season_length,
            self.seasonality,
        ).build_network()
        x_train, y_train = self._get_windows(self._y)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        data = TensorDataset(x_train, y_train)
        loader = DataLoader(data, self.batch_size, shuffle=True)
        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()
        self.network.train()
        for i in range(self.epoch):
            self._run_epoch(i, loader)

        return self

    def _predict(self, X=None, fh=None):
        """
        Predict with fitted model.

        Parameters
        ----------
        X:  Optional, If X is not provided then forecast
            is made on the fitted series.

        fh: Forecasting horizon,
            not used since, forecasting horizon at time of
            fitting is used (direct mode only)
        """
        import torch

        self.network.eval()
        if X is None:
            index = self._fh.to_absolute(self._y.index[-1]).to_numpy()
            column = (
                list(self._y.columns) if isinstance(self._y, pd.DataFrame) else None
            )
            input = self._y[-self.window :]
            input = torch.FloatTensor(np.array(input))
            input = input.unsqueeze(0)
        else:
            index = self._fh.to_absolute(X.index[-1]).to_numpy()
            column = list(X.columns) if isinstance(X, pd.DataFrame) else None
            input = torch.FloatTensor(np.array(X[-self.window :]))
            input = input.unsqueeze(0)
        with torch.no_grad():
            prediction = self.network(input)
            return pd.DataFrame(prediction.squeeze(0), index=index, columns=column)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "input_shape": 1,
            "hidden_size": 1,
            "num_layer": 1,
            "season_length": 2,
            "seasonality": "single",
            "window": 3,
            "stride": 1,
            "batch_size": 32,
            "epoch": 50,
            "optimizer": "Adam",
            "criterion": "mse",
        }
        return [params1, params2]
