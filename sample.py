import numpy as np
import pandas as pd
from math import atan2, sqrt

# Sample points (you can change these values)
t = np.array([0.0, 1.0, 2.0, 3.0])   # time index (uniform spacing dt = 1)
x = np.array([1.0, 2.0, 3.0, 4.0])   # x coordinates
y = np.array([0.0, 1.0, 0.0, -1.0])  # z coordinates (we call it y here)

dt = t[1] - t[0]
eps = 1e-8  # small number for guarding divisions

# First and second derivatives
x_dot = np.zeros_like(x)
y_dot = np.zeros_like(y)
x_dd = np.zeros_like(x)
y_dd = np.zeros_like(y)

# Central differences for interior points
for i in range(1, len(t)-1):
    x_dot[i] = (x[i+1] - x[i-1]) / (2*dt)
    y_dot[i] = (y[i+1] - y[i-1]) / (2*dt)
    x_dd[i] = (x[i+1] - 2*x[i] + x[i-1]) / (dt**2)
    y_dd[i] = (y[i+1] - 2*y[i] + y[i-1]) / (dt**2)

# Forward/backward differences for endpoints
x_dot[0] = (x[1] - x[0]) / dt
y_dot[0] = (y[1] - y[0]) / dt
x_dd[0] = (x[2] - 2*x[1] + x[0]) / (dt**2)
y_dd[0] = (y[2] - 2*y[1] + y[0]) / (dt**2)

x_dot[-1] = (x[-1] - x[-2]) / dt
y_dot[-1] = (y[-1] - y[-2]) / dt
x_dd[-1] = (x[-1] - 2*x[-2] + x[-3]) / (dt**2)
y_dd[-1] = (y[-1] - 2*y[-2] + y[-3]) / (dt**2)

# dy/dx and d²y/dx²
dydx = np.full_like(x, np.nan)
d2ydx2 = np.full_like(x, np.nan)

for i in range(len(t)):
    if abs(x_dot[i]) > eps:
        dydx[i] = y_dot[i] / x_dot[i]
        d2ydx2[i] = (y_dd[i] * x_dot[i] - y_dot[i] * x_dd[i]) / (x_dot[i]**3)
    else:
        dydx[i] = np.nan
        d2ydx2[i] = np.nan

# Polar coordinates
r = np.sqrt(x**2 + y**2)
theta = np.array([atan2(yv, xv) for xv, yv in zip(x, y)])

# Polar rates
rdot = np.full_like(r, np.nan)
thetadot = np.full_like(r, np.nan)
for i in range(len(t)):
    if r[i] > eps:
        rdot[i] = (x[i] * x_dot[i] + y[i] * y_dot[i]) / r[i]
        thetadot[i] = (x[i] * y_dot[i] - y[i] * x_dot[i]) / (r[i]**2)

# Arc length
dx = np.diff(x)
dy = np.diff(y)
segment_lengths = np.sqrt(dx**2 + dy**2)
arc_length = np.sum(segment_lengths)

# Polar area (if theta monotone)
dtheta = np.diff(theta)
is_monotone_increasing = np.all(dtheta > -eps) and np.any(dtheta > eps)
is_monotone_decreasing = np.all(dtheta < eps) and np.any(dtheta < -eps)
polar_area = None
if is_monotone_increasing or is_monotone_decreasing:
    polar_area = 0.5 * np.sum(r[:-1]**2 * dtheta)

# Results in a DataFrame
df = pd.DataFrame({
    "t": t,
    "x": x,
    "y": y,
    "x'": x_dot,
    "y'": y_dot,
    "x''": x_dd,
    "y''": y_dd,
    "dy/dx": dydx,
    "d2y/dx2": d2ydx2,
    "r": r,
    "theta (rad)": theta,
    "r_dot": rdot,
    "theta_dot": thetadot
})
