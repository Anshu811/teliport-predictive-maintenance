"""MLflow pyfunc wrapper: the registered model is the WHOLE pipeline (raw sensors in -> RUL range out)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc


class RULPipelineModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.prep = joblib.load(context.artifacts["preprocessor"])
        self.model = joblib.load(context.artifacts["rul_model"])
        self.unc = joblib.load(context.artifacts["uncertainty"])

    def predict(self, context, model_input, params=None):
        from features import last_cycle
        last = last_cycle(self.prep.transform(model_input))
        pred = np.clip(self.model.predict(last[self.prep.feature_names_]), 0, None)
        lower, upper = self.unc.interval(pred)
        return pd.DataFrame({"unit": last["unit"].to_numpy(), "pred_RUL": np.round(pred, 1),
                             "rul_lower": np.round(lower, 1), "rul_upper": np.round(upper, 1)})
