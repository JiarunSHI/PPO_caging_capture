"""Closed-form UR5e inverse kinematics used to seed exact-MJCF IK.

The constants and branches are a NumPy translation of the BSD-licensed
ROS-Industrial ``ur_kinematics`` implementation. The analytic model and the
rounded MuJoCo Menagerie geometry differ slightly, so every solution is only a
seed and is polished against the compiled MJCF before acceptance.

Source: https://github.com/ros-industrial/universal_robot/blob/noetic-devel/ur_kinematics/src/ur_kin.cpp
"""

from typing import List

import numpy as np


_ZERO = 1e-8
_TWO_PI = 2.0 * np.pi
_D1, _A2, _A3 = 0.1625, -0.425, -0.3922
_D4, _D5, _D6 = 0.1333, 0.0997, 0.0996


def _sign(value: float) -> float:
    return float(int(value > 0.0) - int(value < 0.0))


def _unit_clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def inverse_ur5e(transform_base_flange: np.ndarray, q6_des: float = 0.0) -> np.ndarray:
    """Return all finite analytic branches as rows in ``[0, 2*pi)``."""

    transform = np.asarray(transform_base_flange, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return np.empty((0, 6), dtype=float)
    flat = transform.reshape(-1)
    t02, t00, t01, t03 = -flat[0], flat[1], flat[2], -flat[3]
    t12, t10, t11, t13 = -flat[4], flat[5], flat[6], -flat[7]
    t22, t20, t21, t23 = flat[8], -flat[9], -flat[10], flat[11]

    a_value = _D6 * t12 - t13
    b_value = _D6 * t02 - t03
    radius_squared = a_value * a_value + b_value * b_value
    if abs(a_value) < _ZERO:
        if abs(b_value) < _ZERO:
            return np.empty((0, 6), dtype=float)
        divisor = (-_sign(_D4) * _sign(b_value) if abs(abs(_D4) - abs(b_value)) < _ZERO else -_D4 / b_value)
        angle = float(np.arcsin(_unit_clip(divisor)))
        q1 = [angle % _TWO_PI, (np.pi - angle) % _TWO_PI]
    elif abs(b_value) < _ZERO:
        divisor = (_sign(_D4) * _sign(a_value) if abs(abs(_D4) - abs(a_value)) < _ZERO else _D4 / a_value)
        angle = float(np.arccos(_unit_clip(divisor)))
        q1 = [angle, (_TWO_PI - angle) % _TWO_PI]
    elif _D4 * _D4 > radius_squared + _ZERO:
        return np.empty((0, 6), dtype=float)
    else:
        angle = float(np.arccos(_unit_clip(_D4 / np.sqrt(radius_squared))))
        offset = float(np.arctan2(-b_value, a_value))
        q1 = [(angle + offset) % _TWO_PI, (-angle + offset) % _TWO_PI]

    solutions: List[np.ndarray] = []
    for shoulder in q1:
        c1, s1 = np.cos(shoulder), np.sin(shoulder)
        numerator = t03 * s1 - t13 * c1 - _D4
        if abs(numerator) > abs(_D6) + 1e-7:
            continue
        wrist_angle = float(np.arccos(_unit_clip(numerator / _D6)))
        for wrist_two in (wrist_angle, (_TWO_PI - wrist_angle) % _TWO_PI):
            c5, s5 = np.cos(wrist_two), np.sin(wrist_two)
            if abs(s5) < _ZERO:
                wrist_three = float(q6_des) % _TWO_PI
            else:
                wrist_three = float(np.arctan2(
                    _sign(s5) * -(t01 * s1 - t11 * c1),
                    _sign(s5) * (t00 * s1 - t10 * c1),
                )) % _TWO_PI
            c6, s6 = np.cos(wrist_three), np.sin(wrist_three)
            x04x = -s5 * (t02 * c1 + t12 * s1) - c5 * (
                s6 * (t01 * c1 + t11 * s1) - c6 * (t00 * c1 + t10 * s1)
            )
            x04y = c5 * (t20 * c6 - t21 * s6) - t22 * s5
            p13x = (_D5 * (s6 * (t00 * c1 + t10 * s1) + c6 * (t01 * c1 + t11 * s1))
                    - _D6 * (t02 * c1 + t12 * s1) + t03 * c1 + t13 * s1)
            p13y = t23 - _D1 - _D6 * t22 + _D5 * (t21 * c6 + t20 * s6)
            c3 = (p13x * p13x + p13y * p13y - _A2 * _A2 - _A3 * _A3) / (2.0 * _A2 * _A3)
            if abs(c3) > 1.0 + 1e-7:
                continue
            c3 = _unit_clip(c3)
            elbow_angle = float(np.arccos(c3))
            denominator = _A2 * _A2 + _A3 * _A3 + 2.0 * _A2 * _A3 * c3
            if abs(denominator) < _ZERO:
                continue
            s3 = np.sin(elbow_angle)
            arm_a, arm_b = _A2 + _A3 * c3, _A3 * s3
            q2_options = (
                np.arctan2((arm_a * p13y - arm_b * p13x) / denominator,
                           (arm_a * p13x + arm_b * p13y) / denominator),
                np.arctan2((arm_a * p13y + arm_b * p13x) / denominator,
                           (arm_a * p13x - arm_b * p13y) / denominator),
            )
            q3_options = (elbow_angle, (_TWO_PI - elbow_angle) % _TWO_PI)
            for q2, q3 in zip(q2_options, q3_options):
                c23, s23 = np.cos(q2 + q3), np.sin(q2 + q3)
                q4 = np.arctan2(c23 * x04y - s23 * x04x,
                                x04x * c23 + x04y * s23)
                solution = np.asarray(
                    [shoulder, q2, q3, q4, wrist_two, wrist_three], dtype=float
                ) % _TWO_PI
                if np.all(np.isfinite(solution)):
                    solutions.append(solution)
    if not solutions:
        return np.empty((0, 6), dtype=float)
    return np.unique(np.round(np.asarray(solutions), decimals=12), axis=0)


def wrap_solutions_to_limits(solutions, lower, upper, reference) -> np.ndarray:
    """Choose each equivalent 2*pi branch inside limits nearest reference."""

    lower, upper = np.asarray(lower), np.asarray(upper)
    reference = np.asarray(reference)
    wrapped = []
    for solution in np.asarray(solutions, dtype=float):
        adjusted = np.empty(6, dtype=float)
        for index, angle in enumerate(solution):
            candidates = angle + _TWO_PI * np.arange(-2, 3, dtype=float)
            candidates = candidates[(candidates >= lower[index] - 1e-10) & (candidates <= upper[index] + 1e-10)]
            if not len(candidates):
                break
            adjusted[index] = candidates[np.argmin(np.abs(candidates - reference[index]))]
        else:
            wrapped.append(adjusted)
    return np.asarray(wrapped, dtype=float) if wrapped else np.empty((0, 6), dtype=float)
