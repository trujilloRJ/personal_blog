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
4. [Validation setup](#validation)
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

where \\( \bold{q} = diag([q_v, q_\omega]) \\) and

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
  $$ \bold{\hat{x}\_{k+1}}=f(\bold{x_k}) $$
  $$ \bold{\hat{P}\_{k+1}}=\bold{\dot{F}\_k}\bold{P_k}\bold{\dot{F}^T_k} + \bold{Q_k} $$
- Update:
  $$ \bold{\hat{z}_{k+1}} = h(\bold{\hat{x}\_{k+1}}) $$

  $$ \bold{S_k} = \bold{\dot{H}\_k}\bold{\hat{P}\_{k+1}}\bold{\dot{H}^T_k} + \bold{R} $$
  
  $$ \bold{K_k} = \bold{\hat{P}\_{k+1}}\bold{\dot{H}^T_k}\bold{S_k}^{-1} $$

  $$ \bold{x_{k+1}} = \bold{\hat{x}\_{k+1}} + \bold{K_k}( \bold{z_{k+1}} - \bold{\hat{z}_{k+1}} )  $$

  $$ \bold{P_{k+1}} = (\bold{I_{n_x}} - \bold{K_k}\bold{\dot{H}\_k})\bold{\hat{P}\_{k+1}} $$
  
Where:

- \\( \bold{\dot{F}\_k} \\) and \\( \bold{\dot{H}\_k} \\) are the Jacobian matrices of the state and measurement functions respectively,

- \\( \bold{S_k}  \\) is the innovation covariance,

- \\( \bold{K_k}  \\) is the Kalman gain, 

- and \\( \bold{I_{n_x}} \\) is the identity matrix for the dimension of the state, \\( n_x = 5 \\), in the specific case of CTRV.

An important note is that using the Jacobian, the EKF is aproximating the non-linear function using only the first order term of the Taylor series expansion. Other, more accurate, implementations of the EKF, incorporates higher order terms (REFERENCE) but its implementation is unfeasible for most practical systems.

#### UKF

The UKF tackles nonlinearity in a totally different way. Instead of trying to approximate the non-linear system, it uses a method called the Unscented Transform (UT). This involves picking a set of special points, called sigma points, to represent the spread of possible states. These points are propagated through the non-linear equations, and from the results, the UKF estimates the new mean and covariance based on those transformed points.

Instead of jumping directly into the UKF equations, it is convenient to first state the steps involve in the UT:

1. Generating sigma points:

$$ \mathcal{X_i} $$


![KF example GIF](/posts/images/KF_example.gif)

{{< rawhtml >}}

<iframe src="/posts/images/rmse_comparison.html" width=800 height=800 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

$$\lim_{x \to 0^+} \dfrac{1}{x} = \infty$$

# Conclusions

# Future steps and remaining questions
