+++
title = "EKF vs. UKF performance and robustness comparison for non-linear state estimation"
date = "2024-06-25T15:06:40+02:00"
author = "Javier Trujillo Rodriguez"
authorTwitter = "" #do not include @
cover = ""
tags = ["kalman-filters", "tracking", "numpy"]
keywords = ["cnn", "convolutional neural network", "fmcw radar", "range-Doppler map", "PyTorch"]
description = "A performance comparison between the Extended Kalman Filter (EKF) and the Unscented Kalman Filter (UKF) in a simulated environment"
showFullContent = false
readingTime = false
hideComments = false
math = true
+++

# Motivation

The Kalman Filter (KF) is used for estimating the state of a dynamic system from a series of noisy measurements. KF guarantees an optimal estimation if these two conditions are met:

- **Linear state-space models**: The system state transition (the \\(F\\) matrix) and state-to-measurement function (the \\(H\\) matrix) are linear.

- **Gaussian errors**: The process and measurement noise (\\(Q\\) and \\(R\\) matrices) can be modelled as Gaussian distributions.

In real systems, these conditions are rarely met. In particular, for non-linear systems, the community had proposed variations of the original KF that are **sub-optimal** but still provide good performance. Among the proposed solutions, two approaches have been widely adopted, the Extended-KF (EKF) and the Unscented-KF (UKF).

The EKF approximates non-linear functions by linearizing them around the current estimate. In practice, due to complexity and computational cost, only the first-order Taylor series expansion term is used for the linearization. In contrast, the UKF uses estimates the mean and covariance of the state distribution by selecting a minimal set of points (sigma points) from the current state, propagating these points through the non-linear system, and using the resulting transformed points to calculate the posterior mean and covariance.

As engineers facing a decision between these two, we would like to have insight into the following questions:

- Which filter offer a better overall performance?

- Which filter is computationally cheaper?

- Which filter is more robust to model mismatches?

- Are the filters numerically stable?

This post explore each one of these points by comparing the EKF vs. UKF in a simulated environment assuming a target moving with Constant Turn Rate and Velocity (CTRV) and a sensor that measures range and bearing. Both the motion model and the measurement function are non-linear.

## Table of Contents

1. [Motion model and measurement function](#motion-model-and-measurement-function)
2. [EKF and UKF algorithm](#ekf-and-ukf-algorithm)
3. [Implementation](#example2)
4. [Simulation and validation metrics](#validation)
5. [Results](#Results)
6. [Conclusions](#Conclusions)

# Motion model and measurement function

#### CTRV motion model

We will consider a target moving on a 2D-plane whose motion can be modelled with the CTRV. The filter state is represented in Fig. 1 and defined in polar coordinates as:

$$\bold{x_k}=[x_k, y_k, v_k, \phi_k, \omega_k]^T$$

Where the subscript \\(k\\) represents the time index and:

- \\(x\\) and \\(y\\) is the target position in the X and Y axis respectively,
- \\(\phi\\) is the target heading with respect to X axis,
- \\(v\\) is the target velocity along the direction of the headin,
- and \\(\omega\\) is the target turn rate which represents the heading rate of change.

{{< rawhtml >}}
<image src="/posts/images/02_state.svg" alt="state and measurement model" position="center" style="border-radius: 8px; width: 550px; height: 320px; object-fit: cover; object-position: top;">
{{< /rawhtml >}}
Fig.1 Target state with CTRV and sensor measurement.

This motion model assumes that both the target velocity and turn rate are constant, the state transition function can be obtained as:

$$
\bold{\hat{x}_{k+1}}=f(\bold{x_k})=
\begin{bmatrix}
x_k + v_k/\omega_k ( \sin(\phi_k + \omega_kT) - \sin(\phi_k)) \\\
y_k + v_k/\omega_k ( -\cos(\phi_k + \omega_kT) + \cos(\phi_k)) \\\
\phi_k + \omega_kT \\\
v_k \\\
\omega_k
\end{bmatrix}
$$

As can be seen this function is non-linear. This means that if we consider that the state distribution is gaussian at time \\(k\\) after passing it through the function the resulting distribution is no longer gaussian. This violates the assumptions made by the standard KF and if applied it will result in filter divergence. To deal with this, the research community had proposed several solutions, by far the most widely adopted in the industry is the EKF. As mentioned before the implementation of the EKF require the computation of Jacobians for the state transition function, to avoid cluttered the content, it has been defined in the Appendix section.

#### Range and bearing measurement function

As visible in Fig. 1, the sensor provides at each time \\(k\\) the target position by measuring range \\(r^m_k\\) and bearing \\(\theta^m_k\\) defining the measurement:

$$\bold{z_k}=[r^m_k, \theta^m_k]^T$$

However, the track states contains the target position in cartesian coordinates. Therefore, to update the filter state, we need a measurement function that maps between the state space and measurement space:

$$
\bold{\hat{z}_{k+1}}=h(\bold{\hat{x}})=
\begin{bmatrix}
\sqrt{x_k^2 + y_k^2} \\\
\tan^{-1}({y_k}/{x_k})
\end{bmatrix}
$$

It is clear that the measurement function is also non-linear. The Jacobian of this function, required for the EKF, can also be found in the Appendix section.

#### Modelling measurement and process noise

Due to noise, the sensor never provides an exact measurement. The noise is usually modeled as independent random gaussian distributions with zero mean and covariance defined as:

$$
\bold{R}=
\begin{bmatrix}
\sigma^2_r & 0 \\\
0 & \sigma^2_{\theta}
\end{bmatrix}
$$

Where \\(\sigma_r\\) and \\(\sigma\_{\theta}\\) are the noise standard deviation in range and bearing, respectively. These values, are usually intrinsic to the sensor characteristics. For example in FMCW radars, the error in range is related to the range resolution, which in turn depends on the waveform bandwidth.

In addition, we also need to account for errors in our motion model. In practice, no target will move exactly following CTRV, there will be maneuvers and slight path deviations. We modelled this by introducing process noise as zero-mean gaussian distributions to the target velocity \\(q_v\\) and turn rate \\(q\_{\omega}\\). The covariance matrix is obtained as

$$
\bold{Q_k}= \bold{\Gamma_k}\bold{q}\bold{\Gamma_k^T}
$$

where \\( \bold{q} = diag([q_v^2, q_\omega^2]) \\) and

$$
\bold{\Gamma_k}=
\begin{bmatrix}
\cos(\phi_k)T^2/2 & 0 \\\
0 & \sin(\phi_k)T^2/ 2 \\\
0 & T^2/2 \\\
T & 0 \\\
0 & T
\end{bmatrix}
$$

# EKF and UKF algorithm

#### EKF

As mentioned before, the EKF deal with system non-linearities by linerizing them around the current estimate. In practice, the difference with respect to the standard KF is that the matrices used for covariance propagation are the Jacobian of the state transition and measurement function. The EKF cycle can be described by the following equations:

- Prediction:
  $$ \bold{\hat{x}_{k+1}}=f(\bold{x_k}) $$

  $$ \bold{\hat{P}_{k+1}}=\bold{\dot{F}_k}\bold{P_k}\bold{\dot{F}^T_k} + \bold{Q_k} $$

- Update:

  $$ \bold{\hat{z}\_{k+1}} = h(\bold{\hat{x}_{k+1}}) $$

  $$ \bold{S_k} = \bold{\dot{H}\_k}\bold{\hat{P}_{k+1}}\bold{\dot{H}^T_k} + \bold{R} $$
  
  $$ \bold{K_k} = \bold{\hat{P}_{k+1}}\bold{\dot{H}^T_k}\bold{S_k}^{-1} $$

  $$ \bold{x_{k+1}} = \bold{\hat{x}\_{k+1}} + \bold{K_k}( \bold{z_{k+1}} - \bold{\hat{z}_{k+1}} )  $$

  $$ \bold{P_{k+1}} = (\bold{I_{n_x}} - \bold{K_k}\bold{\dot{H}_k})\bold{\hat{P}\_{k+1}} $$
  
Where:

- \\( \bold{\dot{F}\_k} \\) and \\( \bold{\dot{H}\_k} \\) are the Jacobian matrices of the state and measurement functions respectively,

- \\( \bold{S_k}  \\) is the innovation covariance,

- \\( \bold{K_k}  \\) is the Kalman gain, 

- and \\( \bold{I_{n_x}} \\) is the identity matrix for the dimension of the state, \\( n_x = 5 \\), in the specific case of CTRV.

An important note is that using the Jacobian, the EKF is aproximating the non-linear function using only the first order term of the Taylor series expansion. Other, more accurate, implementations of the EKF, incorporates higher order terms (REFERENCE) but its implementation is unfeasible for most practical systems.

#### UKF

The UKF tackles nonlinearity in a totally different way. Instead of trying to approximate the non-linear system, it uses a method called the Unscented Transform (UT). This involves picking a set of special points, called sigma points, to represent the spread of possible states. These points are propagated through the non-linear function, and from the results, the UKF estimates the new mean and covariance based on those transformed points.

Instead of jumping directly into the UKF equations, it is convenient to first go through the Unscented Transform. In this context, the UT can be defined as an operation that takes as input the state mean \\( \bold{x_k} \\), covariance \\( \bold{P_k} \\) and the non-linear function \\( f \\). Then, the UT estimates the mean \\( \bold{y_m} \\) and covariance \\( \bold{P_y} \\) of the resulting distribution after passing the state through the non-linear function. In addition, the propagated sigma points \\( \mathcal{Y} \\) are also of interest for UKF.

$$ (\bold{y_m},\bold{P_y}, \mathcal{Y}) = UT(f, \bold{x_k}, \bold{P_k}) $$

Step by step, the UT is doing the following:

1 Selecting parameters:

  $$\alpha = 1, \hspace{0.2cm} \beta = 2, \hspace{0.2cm} \kappa = 0$$

  $$\lambda = \alpha^2(n_x+\kappa)-n_x$$

2 Computing weights

$$ w^m_0 = \frac{\lambda}{\lambda+n_x}, \hspace{0.5cm} w^c_0 = \frac{\lambda}{\lambda+n_x} + 1 - \alpha^2 + \beta $$

$$ w^m_i = w^c_i = \frac{1}{2(n_x+\lambda)} $$

3 Generating sigma points set \\( \mathcal{X} \\):

$$ \mathcal{X_0} = \bold{x_k} $$

$$ \mathcal{X_i} = \bold{x_k} + \left(\sqrt{(\lambda+n_x)P_k}\right)_i, \hspace{0.5cm} i=1,..,n_x $$

$$ \mathcal{X_i} = \bold{x_k} - \left(\sqrt{\lambda+n_x}\right)_{i-n_x}, \hspace{0.5cm} i=n_x + 1,..,2n_x $$

4 Propagating sigma points through non-linear function (either  \\( f(.) \\) or \\( h(.) \\))

$$ \mathcal{Y_i} = f(\mathcal{X_i}), \hspace{0.5cm} i=0,...,2n_x$$

5 Estimating mean and covariance of the propagated distribution

$$ \bold{y_m} = \sum_{i=0}^{2n_x}w^m_i\mathcal{Y_i} $$

$$ \bold{P_y} = \sum_{i=0}^{2n_x}w^c_i(\mathcal{Y_i} - \bold{x}_m)(\mathcal{Y_i} - \bold{x}_m)^T $$

Where \\( n_x \\) is the dimension of  the state and \\( \bold{L_k} = \sqrt{(\lambda+n_x)\bold{P_k}} \\) is the scaled square root of the state covariance matrix. The square root of a matrix is not defined, conventionally people use Cholesky factorization as an equivalent to matrix square root and that is what we used in the UKF implementation by leveraging `scipy.linalg.cholesky` function.

Finally, the UKF cyclic operation can be defined as:

- Prediction

$$ \left(\bold{\hat{x}_{k+1}},\bold{P_y},\mathcal{Y}\right) = UT(f, \bold{x_k}, \bold{P_k}) $$

$$ \bold{\hat{P}_{k+1}} = \bold{P_y} + \bold{Q_k}$$

- Update

$$ (\bold{\hat{z}\_{k+1}},\bold{P_{zz}},\mathcal{Z}) = UT(h, \bold{\hat{x}\_{k+1}}, \bold{\hat{P}\_{k+1}}) $$

$$ \bold{P_{xz}} = \sum_{i=0}^{2n_x}w^c_i(\mathcal{Y_i} - \bold{\hat{x}\_{k+1}})(\mathcal{Z_i} - \bold{\hat{z}_{k+1}})^T $$

$$ \bold{K_k} = \bold{P_{xz}}\bold{P^{-1}_{zz}} $$

$$ \bold{x\_{k+1}} = \bold{\hat{x}\_{k+1}} + \bold{K_k}( \bold{z_{k+1}} - \bold{\hat{z}_{k+1}} )  $$

$$ \bold{P_{k+1}} = \bold{P_{k+1}} -  \bold{K_k}\bold{P_{zz}}\bold{K_k^T}$$

It is important to remark that the UKF does not require the computation of Jacobians as opposed to EKF. Despite this, it has been proven that UKF is able to achieve approximations of 3rd order for gaussian cases [REFERENCE].

![KF example GIF](/posts/images/KF_example.gif)

# Implementation

The implementation of both filters can be found in `filters.py` module. The EKF under `EKF_CTRV` class and the UKF under `UKF_CTRV`. Notice that the motion models and measurement function are injected as dependencies of each filter. This enable more flexibility in the future in case we want to try different motion models and measurement functions.

A note on the CTRV implementation. From the transition function, it is visible that the model becomes undefined when target is not turning \\( \omega=0 \\). For those cases, both the transition function and its Jacobian are simplified to Constant Velocity (CV) in the code.

#### Note on angle arithmetics

Angles are modular quantities and as such, additional checks needs to be in-place when dealing with them. For instance, we cannot compute directly the average of a set of angles: the mean angle between 359\\( \degree \\) and 1\\( \degree \\) is 0\\( \degree \\), not 180\\( \degree \\). Instead, we need to use the [circular mean](https://en.wikipedia.org/wiki/Circular_mean).

$$ \bar{\theta} = \tan^{-1}\left( \frac{\sum_{i=0}^{N-1} \sin(\theta_i)}{\sum_{i=0}^{N-1} \cos(\theta_i)} \right) $$

In our case, we have the bearing angle in the measurement \\( \theta \\) and heading angle in the state \\( \phi \\). We need to ensure that (i) angles are always in the range between \\( (-\pi, \pi) \\) and (ii) use the *weighted* circular mean when estimating state mean in UKF. In the same `filters.py` module you will find the corresponding implementation.

# Simulation and validation metrics

A simulated target is generated moving in a straight line at constant velocity, then making a sharp turn and finally moving straight again. A sensor provides noisy measurement of range and bearing of the target at each frame. The following table summarizes the parameters used for the simulation, including the sensor errors:

| Parameter        | Symbol | Value     |
| ---------------- | ------ | --------  |
| Time between frames | \\( T \\) | 50 ms     |
| Iterations | \\( N \\) | 10     |
| Sensor range standard deviation | \\( \sigma_r \\) | 0.5 m     |
| Sensor azimuth standard deviation | \\( \sigma_\theta \\) | 2\\( \degree \\)     |
| Process noise velocity | \\( q_v \\) | 5 m/s     |
| Process noise turn rate | \\( q_\omega \\) | 10\\( \degree \\)     |

All the logic for the simulation is in `simulation.py` module. The following figure shows an example of one iteration. It is important to notice how the position errors in X and Y increase as the target moves further from the sensor. This is expected due to the nonlinear nature of coordinate conversion from polar to cartesian.

[INCLUDE FIGURE]

The metric selected to evaluate the performance of the filters is the classical Root Mean Squared Error (RMSE) defined as:

$$ RMSE(\bold{x}) = \epsilon_x = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(\hat{x}_i-x_t)^2} $$

Where \\( N \\) is the number of iterations, \\( \hat{x}_i \\) is the filter estimation at the i-th iteration and \\( x_t \\) is the true value.

# Results

# Conclusions

# Future steps and remaining questions

{{< rawhtml >}}
<iframe src="/posts/images/rmse_comparison.html" width=800 height=800 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

