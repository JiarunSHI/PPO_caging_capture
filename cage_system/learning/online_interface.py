from collections import deque
from typing import Optional

import numpy as np

from cage_system.learning.dataset import make_cage_input
from cage_system.learning.history_quality import (
    HistoryQualityResult,
    evaluate_history_quality,
)
from cage_system.trajectory.online import OnlineTrajectoryEstimator
from cage_system.types import PointObservation


class HistoricalCageInference:
    """Predict current t from five prior [x,y,z,speed] frames, then append t."""

    def __init__(
        self,
        frozen_network,
        trajectory_config,
        kalman_config,
        timing_config,
        history_quality_config,
    ):
        self.network = frozen_network
        runtime_period_s = 1.0 / float(timing_config["detection_rate_hz"])
        checkpoint_period_s = getattr(
            frozen_network, "nominal_detection_period_s", None
        )
        if checkpoint_period_s is None or not np.isclose(
            float(checkpoint_period_s), runtime_period_s, rtol=1e-6, atol=1e-9
        ):
            raise ValueError(
                "Cage checkpoint detection period does not match runtime: "
                "checkpoint={} s, runtime={:.6f} s".format(
                    checkpoint_period_s, runtime_period_s
                )
            )
        checkpoint_quality = getattr(frozen_network, "history_quality_config", None)
        if checkpoint_quality is not None:
            for key, value in history_quality_config.items():
                if key not in checkpoint_quality or not np.isclose(
                    float(checkpoint_quality[key]), float(value)
                ):
                    raise ValueError(
                        "Cage checkpoint and runtime history-quality thresholds differ"
                    )
        checkpoint_kalman = getattr(frozen_network, "kalman_config", None)
        if checkpoint_kalman is None:
            raise ValueError("Cage checkpoint is missing its visual Kalman contract")
        for key, value in checkpoint_kalman.items():
            if key not in kalman_config or not np.isclose(
                float(kalman_config[key]), float(value)
            ):
                raise ValueError(
                    "Cage checkpoint and runtime visual Kalman settings differ"
                )
        self.estimator = OnlineTrajectoryEstimator(trajectory_config, kalman_config, timing_config)
        self.timing_config = timing_config
        self.history_quality_config = history_quality_config
        self.history = deque(maxlen=5)
        self.last_target: Optional[np.ndarray] = None
        self.last_target_timestamp: Optional[float] = None
        self.last_quality = HistoryQualityResult(
            False,
            "insufficient_five_frame_history",
            5,
            float("inf"),
            float("inf"),
            float("inf"),
        )

    def predict_current_then_update(self, current_observation: PointObservation):
        prediction = None
        quality = evaluate_history_quality(
            list(self.history),
            current_observation.timestamp,
            self.timing_config,
            self.history_quality_config,
        )
        self.last_quality = quality
        if quality.valid:
            input_matrix = make_cage_input(list(self.history))
            prediction = self.network.predict(input_matrix)
            self.last_target = prediction.copy()
            self.last_target_timestamp = float(current_observation.timestamp)
        if not current_observation.valid and self.estimator.filter.timestamp is None:
            # A missing first detection cannot initialize a position filter and
            # therefore must not create a fabricated history row.
            return prediction
        current_frame = self.estimator.update(current_observation)
        self.history.append(current_frame)
        return prediction

    def update_then_predict(self, completed_observation: PointObservation):
        """Append one completed frame, then predict from the latest five rows.

        PPO privileged initialization uses the third completed five-frame
        window and begins policy collection from the fourth.  No unobserved
        next object point is needed: the prediction timestamp is simply the
        next configured visual instant implied by the completed history.
        """
        if not completed_observation.valid and self.estimator.filter.timestamp is None:
            self.last_quality = HistoryQualityResult(
                False,
                "insufficient_five_frame_history",
                5,
                float("inf"),
                float("inf"),
                float("inf"),
            )
            return None
        current_frame = self.estimator.update(completed_observation)
        self.history.append(current_frame)
        prediction_timestamp = float(completed_observation.timestamp) + 1.0 / float(
            self.timing_config["detection_rate_hz"]
        )
        quality = evaluate_history_quality(
            list(self.history),
            prediction_timestamp,
            self.timing_config,
            self.history_quality_config,
        )
        self.last_quality = quality
        if not quality.valid:
            return None
        prediction = self.network.predict(make_cage_input(list(self.history)))
        self.last_target = prediction.copy()
        self.last_target_timestamp = prediction_timestamp
        return prediction

    @property
    def ready(self):
        return bool(self.last_quality.valid)
