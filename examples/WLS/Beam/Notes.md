# Context and Physical System

This experiment models a simple two-mode oscillator, analogous to a vibrating beam with two dominant flexural modes. In structural dynamics, such a beam can oscillate in several shapes (modes), each with its own frequency and damping. Sensors (such as accelerometers or displacement probes) are placed at different positions along the beam, and each sensor measures a linear combination of the modal displacements, weighted by the mode shape at the sensor's location.
#### Physical Interpretation
 
The modal coordinates $q_1(t)$ and $q_2(t)$ represent the time-dependent amplitudes of the first and second vibration modes, respectively. Each sensor measures a signal $x_j(t)$ given by:
 
$$
x_j(t) = \phi_{1j} \cdot q_1(t) + \phi_{2j} \cdot q_2(t)
$$

where $\phi_{1j}$ and $\phi_{2j}$ are the mode shape weights (sensitivities) of mode 1 and mode 2 at sensor location $j$. This means each sensor sees a different mixture of the two modal responses, depending on its placement.

#### Equations of Motion
 
The modal coordinates evolve according to decoupled second-order differential equations:

$$
\begin{cases}
\ddot{q}_1(t) + 2 \zeta_1 \omega_1 \dot{q}_1(t) + \omega_1^2 q_1(t) = 0 \\
\ddot{q}_2(t) + 2 \zeta_2 \omega_2 \dot{q}_2(t) + \omega_2^2 q_2(t) = 0
\end{cases}
$$

where $\omega_1$, $\omega_2$ are the natural frequencies and $\zeta_1$, $\zeta_2$ are the damping ratios for each mode. Ideally, SINDy should identify these linear second-order ODEs from the measured sensor data.

The second-order system can be rewritten as a first-order system by defining state variables for each mode's displacement and velocity:

$$
\begin{cases}
\dot{q}_1 = v_1 \\
\dot{v}_1 = -2 \zeta_1 \omega_1 v_1 - \omega_1^2 q_1 \\
\dot{q}_2 = v_2 \\
\dot{v}_2 = -2 \zeta_2 \omega_2 v_2 - \omega_2^2 q_2
\end{cases}
$$
  
#### Mode Shape Interpretation

Each coefficient $\phi_{m,j} = \phi_m(r_j)$ represents the *mode shape amplitude* of mode *m* at the location of sensor *j*.  
It quantifies how strongly that mode contributes to the measured displacement at that position.  

For a vibrating beam (e.g., a cantilever), each bending mode has a characteristic spatial profile:

- **Mode 1**: one half-wave, no internal node.  Approximate shape: $\phi_1(r) \approx \sin(\pi r / 2L)$
- **Mode 2**: two half-waves, one internal node near mid-length.  Approximate shape: $\phi_2(r) ≈ \sin(3\pi r / 2L)$

If the beam length is **L**, examples of \phi values are:

| Sensor location (rⱼ / L) | $\phi_1(r_j)$ | $\phi_2(r_j)$ | Interpretation                                |
| ------------------------ | ------------- | ------------- | --------------------------------------------- |
| 0.0 (clamped base)       | 0.00          | 0.00          | No motion (node for all modes)                |
| 0.25                     | 0.38          | 0.71          | Moderate mode 1, strong mode 2                |
| 0.50 (mid-span)          | 0.71          | 0.00          | Strong mode 1, node of mode 2                 |
| 0.75                     | 0.92          | -0.71         | Very strong mode 1, opposite phase for mode 2 |
| 1.0 (free tip)           | 1.00          | 0.00          | Maximum for mode 1, node for mode 2           |

**Example:**  
- A sensor near the *free tip* (r ≈ L) measures mostly mode 1 ⇒ \phi₁ large, \phi₂ small.  
- A sensor near the *mid-span* (r ≈ L/2) sees less mode 1 and almost no mode 2 (since it is near a node).  
- A sensor closer to the *clamped end* (r ≈ 0) records almost no motion.

Hence, the choice of sensor location defines \phiₘⱼ and determines which modal combinations appear in the measured trajectories.

#### Experiment Purpose and Motivation

This experiment serves as a baseline test for system identification using SINDy with synthetic, noise-contaminated data from multiple sensors. By starting with two sensors and two modes, we can assess SINDy's ability to recover the correct underlying dynamics and understand how sensor placement (mode shape weights) affects identification. This sets the stage for more advanced experiments involving more sensors, higher noise levels, and the use of weighting schemes to account for sensor quality and redundancy.
  
#### Steps

1. Define system and simulation parameters.
2. Generate modal coordinates for both modes.
3. Construct sensor signals using mode-shape weights.
4. Add noise to the measurements.
5. Plot the sensor trajectories.
6. Prepare data and fit a SINDy model.

