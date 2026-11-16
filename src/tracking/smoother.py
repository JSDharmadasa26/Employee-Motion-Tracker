"""Directional-innovation adaptive spatial smoothing filters for bounding boxes and 13-keypoint skeletons.

Eliminates zero-mean high-frequency jitter when stationary/typing while locking onto genuine
directional fast movements (doorway walking, rapid posture/arm transitions) with near-zero lag.
"""

from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np


class KalmanPoint2DSmoother:
    """Directional-innovation adaptive 2D constant-velocity Kalman filter for skeletal keypoints.

    State: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    - Zero-mean jitter cancels in directional innovation EMA -> tight Q suppresses flicker (>65%).
    - Sustained directional fast motion accumulates in EMA -> dynamic Q boost locks velocity & position (<3px lag).
    - Barrier occlusion (low conf): confidence-scaled R and gentle coasting preserve anatomical pose.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_accel_std: float = 16.0,
        base_measurement_noise_std: float = 3.6,
    ) -> None:
        self.dt = dt

        # Pure constant-velocity transition during observed tracking (zero steady-state velocity lag)
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        qa = process_accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        self.Q_base = np.array(
            [
                [dt4 * qa, 0.0, dt3 * qa, 0.0],
                [0.0, dt4 * qa, 0.0, dt3 * qa],
                [dt3 * qa, 0.0, dt2 * qa, 0.0],
                [0.0, dt3 * qa, 0.0, dt2 * qa],
            ],
            dtype=np.float64,
        )

        self.base_R_variance = base_measurement_noise_std ** 2
        self.state: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.ema_innov: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.smoothed_conf: float = 0.0
        self.is_initialized: bool = False

    def reset(self) -> None:
        self.state = None
        self.P = None
        self.ema_innov = np.zeros((2, 1), dtype=np.float64)
        self.smoothed_conf = 0.0
        self.is_initialized = False

    def predict_position(self) -> Optional[Tuple[float, float]]:
        """Return next-step predicted (x, y) without mutating filter state."""
        if not self.is_initialized or self.state is None:
            return None
        pred = np.dot(self.F, self.state)
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, x: float, y: float, conf: float) -> Tuple[float, float, float]:
        """Update filter with observation (x, y, conf) and return smoothed (x, y, conf)."""
        valid_obs = conf >= 0.04 and not (abs(x) < 1e-4 and abs(y) < 1e-4)

        if not self.is_initialized:
            self.state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
            self.P = np.diag([9.0, 9.0, 900.0, 900.0]).astype(np.float64)
            self.ema_innov = np.zeros((2, 1), dtype=np.float64)
            self.smoothed_conf = conf
            self.is_initialized = valid_obs
            return float(x), float(y), float(conf)

        # 1. Kinematic prediction
        pred_state = np.dot(self.F, self.state)
        self.smoothed_conf = 0.65 * self.smoothed_conf + 0.35 * conf

        if not valid_obs:
            # Barrier coasting with velocity damping so occluded joints hold natural pose
            self.state = pred_state
            self.state[2:, 0] *= 0.78
            self.ema_innov *= 0.5
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q_base
            self.P = 0.5 * (self.P + self.P.T)
            return float(self.state[0, 0]), float(self.state[1, 0]), max(self.smoothed_conf * 0.88, 0.0)

        # 2. Directional innovation EMA separates zero-mean jitter from genuine rapid motion
        z = np.array([[x], [y]], dtype=np.float64)
        y_residual = z - np.dot(self.H, pred_state)
        self.ema_innov = 0.72 * self.ema_innov + 0.28 * y_residual
        dir_mag = float(np.hypot(self.ema_innov[0, 0], self.ema_innov[1, 0]))

        Q_step = self.Q_base.copy()
        if dir_mag > 9.5 and conf >= 0.30:
            excess = dir_mag - 9.5
            pos_boost = min((excess * 2.2) ** 2, 3000.0)
            vel_boost = min(((excess / max(self.dt, 1e-3)) * 0.85) ** 2, 350000.0)
            Q_step[0, 0] += pos_boost
            Q_step[1, 1] += pos_boost
            Q_step[2, 2] += vel_boost
            Q_step[3, 3] += vel_boost

        P_pred = np.dot(self.F, np.dot(self.P, self.F.T)) + Q_step

        # 3. Confidence-adaptive measurement covariance R (handles barrier-occluded joints smoothly)
        effective_conf = max(conf, 0.08)
        R = np.eye(2, dtype=np.float64) * (self.base_R_variance / (effective_conf ** 2))

        S = np.dot(self.H, np.dot(P_pred, self.H.T)) + R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))

        self.state = pred_state + np.dot(K, y_residual)
        # Update directional EMA post-fit so locked velocity immediately quiets excess boost
        post_res = z - np.dot(self.H, self.state)
        self.ema_innov = 0.50 * self.ema_innov + 0.50 * post_res

        I_mat = np.eye(4, dtype=np.float64)
        self.P = np.dot((I_mat - np.dot(K, self.H)), P_pred)
        self.P = 0.5 * (self.P + self.P.T)

        return float(self.state[0, 0]), float(self.state[1, 0]), float(self.smoothed_conf)


class KalmanBBoxSmoother:
    """Directional-innovation adaptive bounding box Kalman filter tracking [cx, cy, w, h, vcx, vcy, vw, vh]."""

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 2.4,
    ) -> None:
        self.dt = dt
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt
        # Center velocity [vcx, vcy] is undamped during active tracking (zero walking lag);
        # Box size rate [vw, vh] is gently damped so box dimensions don't oscillate.
        self.F[6, 6] = 0.82
        self.F[7, 7] = 0.82

        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        self.Q_base = np.diag([0.02, 0.02, 0.02, 0.02, 3.5, 3.5, 1.5, 1.5]).astype(np.float64) * (
            (process_noise_std / 2.0) ** 2
        )
        self.R_base = np.eye(4, dtype=np.float64) * ((measurement_noise_std * 1.5) ** 2)

        self.state: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.ema_innov_c: np.ndarray = np.zeros((2, 1), dtype=np.float64)
        self.is_initialized: bool = False

    def reset(self) -> None:
        self.state = None
        self.P = None
        self.ema_innov_c = np.zeros((2, 1), dtype=np.float64)
        self.is_initialized = False

    def predict_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Return kinematic predicted [x1, y1, x2, y2] for Hungarian fast-motion matching."""
        if not self.is_initialized or self.state is None:
            return None
        pred = np.dot(self.F, self.state)
        cx, cy = float(pred[0, 0]), float(pred[1, 0])
        w, h = max(float(pred[2, 0]), 1.0), max(float(pred[3, 0]), 1.0)
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    def update(
        self, x1: float, y1: float, x2: float, y2: float, conf: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """Update filter with bounding box [x1, y1, x2, y2] (or coast when conf <= 0)."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        if not self.is_initialized:
            self.state = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
            self.P = np.diag([12.0, 12.0, 12.0, 12.0, 900.0, 900.0, 100.0, 100.0]).astype(np.float64)
            self.ema_innov_c = np.zeros((2, 1), dtype=np.float64)
            self.is_initialized = True
            return x1, y1, x2, y2

        pred_state = np.dot(self.F, self.state)
        if conf <= 0.02:
            self.state = pred_state
            self.state[4:, 0] *= 0.78
            self.ema_innov_c *= 0.5
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q_base
            self.P = 0.5 * (self.P + self.P.T)
        else:
            z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)
            residual = z - np.dot(self.H, pred_state)
            self.ema_innov_c = 0.52 * self.ema_innov_c + 0.48 * residual[:2]
            dir_shift = float(np.hypot(self.ema_innov_c[0, 0], self.ema_innov_c[1, 0]))

            Q_step = self.Q_base.copy()
            if dir_shift > 7.0:
                excess = dir_shift - 7.0
                p_boost = min((excess * 1.4) ** 2, 2500.0)
                v_boost = min(((excess / max(self.dt, 1e-3)) * 0.55) ** 2, 250000.0)
                Q_step[0, 0] += p_boost
                Q_step[1, 1] += p_boost
                Q_step[4, 4] += v_boost
                Q_step[5, 5] += v_boost

            P_pred = np.dot(self.F, np.dot(self.P, self.F.T)) + Q_step
            effective_conf = max(conf, 0.15)
            R_dynamic = self.R_base / (effective_conf ** 2)

            S = np.dot(self.H, np.dot(P_pred, self.H.T)) + R_dynamic
            K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))

            self.state = pred_state + np.dot(K, residual)
            post_res = z[:2] - self.state[:2]
            self.ema_innov_c = 0.50 * self.ema_innov_c + 0.50 * post_res

            I_mat = np.eye(8, dtype=np.float64)
            self.P = np.dot((I_mat - np.dot(K, self.H)), P_pred)
            self.P = 0.5 * (self.P + self.P.T)

        smooth_cx = float(self.state[0, 0])
        smooth_cy = float(self.state[1, 0])
        smooth_w = max(float(self.state[2, 0]), 1.0)
        smooth_h = max(float(self.state[3, 0]), 1.0)

        return (
            smooth_cx - smooth_w / 2.0,
            smooth_cy - smooth_h / 2.0,
            smooth_cx + smooth_w / 2.0,
            smooth_cy + smooth_h / 2.0,
        )


class RollingAverageSmoother:
    """Windowed moving average smoother for 2D points and bounding boxes."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.point_history: Dict[str, deque] = {}
        self.bbox_history: deque = deque(maxlen=window_size)

    def reset(self) -> None:
        self.point_history.clear()
        self.bbox_history.clear()

    def update_point(self, joint_name: str, x: float, y: float, conf: float) -> Tuple[float, float, float]:
        if joint_name not in self.point_history:
            self.point_history[joint_name] = deque(maxlen=self.window_size)

        hist = self.point_history[joint_name]
        hist.append((x, y, conf))

        weights = [max(item[2], 0.05) for item in hist]
        total_w = sum(weights)
        if total_w < 1e-6:
            return x, y, conf

        sx = sum(item[0] * w for item, w in zip(hist, weights)) / total_w
        sy = sum(item[1] * w for item, w in zip(hist, weights)) / total_w
        sc = sum(item[2] for item in hist) / len(hist)
        return sx, sy, sc

    def update_bbox(
        self, x1: float, y1: float, x2: float, y2: float, conf: float = 1.0
    ) -> Tuple[float, float, float, float]:
        if conf > 0.02:
            self.bbox_history.append((x1, y1, x2, y2, conf))

        if not self.bbox_history:
            return x1, y1, x2, y2

        weights = [max(item[4], 0.05) for item in self.bbox_history]
        total_w = sum(weights)
        sx1 = sum(item[0] * w for item, w in zip(self.bbox_history, weights)) / total_w
        sy1 = sum(item[1] * w for item, w in zip(self.bbox_history, weights)) / total_w
        sx2 = sum(item[2] * w for item, w in zip(self.bbox_history, weights)) / total_w
        sy2 = sum(item[3] * w for item, w in zip(self.bbox_history, weights)) / total_w
        return sx1, sy1, sx2, sy2
