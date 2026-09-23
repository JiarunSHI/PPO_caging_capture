# Dynamic Cage validation trajectory protocol

## 1. Fixed evaluation archive

Create one immutable archive of 30 seed-controlled, open trajectories with the existing randomized inference generator. Its master seed must not appear in offline EST data generation, policy training, checkpoint selection, or hyperparameter tuning. The generator remains random inside the configured physical range; speed, curvature, height, Base distance, and motion direction are recorded afterward for subgroup analysis rather than forcing a separate hand-designed benchmark distribution.

All accepted trajectories must satisfy:

- duration: 5--7 s;
- speed: 0.11--0.18 m/s with 10%--20% smooth modulation;
- nominal curvature: 2.2--3.6 1/m, maximum curvature no greater than 8.5 1/m;
- five-frame turning angle: 4--25 degrees;
- total turning angle: no greater than 150 degrees;
- endpoint displacement / arc length: at least 0.45, to reject nearly closed paths;
- Base distance: 0.38--0.70 m;
- height: 0.18--0.72 m;
- valid initial observation pose and no initial kinematic or collision violation.

The accepted trajectory file, seeds, generator configuration, and rejection audit must be frozen before evaluating a controller.

## 2. Trajectory-set quality

Report trajectory coverage independently of controller performance:

1. occupied cells and maximum count imbalance in a speed--curvature--height grid;
2. acceptance rate after reachability, collision, and visibility screening;
3. nearest-neighbor distance from each validation descriptor to the offline generation archive, to detect duplicates and poor distribution separation;
4. maximum one-step curvature increment and five-frame turning-angle distribution, to exclude discontinuous or physically implausible paths;
5. distributions of arc length, total turning, endpoint-displacement ratio, Base distance, and height.

## 3. Tracking quality

For every trajectory, report:

- final outcome: complete, joint-limit termination, visual failure, collision, timeout, or other failure;
- PPO-entry indicator and entry time;
- position and orientation RMSE, median, P95, and maximum after PPO entry;
- fraction of time inside 15 mm / 45 degrees and inside 5 mm / 5 degrees;
- visible-frame fraction and longest consecutive miss run;
- minimum joint-limit margin and action-saturation fraction;
- RMS joint acceleration and jerk;
- center-position and velocity error against hidden simulation truth, used only for evaluation;
- results broken down by speed, curvature, height, and stress subgroup.

Use Wilson 95% confidence intervals for rates. Do not replace the outcome vector with one aggregate score: a high mean score can hide the low/downward and reach-boundary failures that this protocol is intended to reveal.
