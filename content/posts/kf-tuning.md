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

We will consider a target whose motion can be modelled with the CTRV. The filter state is defined in polar coordinates as:

$$\bold{x_k}=[x_k, y_k, v_k, \phi_k, \omega_k]^T$$

Where the subscript \\(k\\) represents the time index and:

- \\(x\\) and \\(y\\) is the target position in the X and Y axis respectively,
- \\(\phi\\) is the target heading with respect to X axis,
- \\(v\\) is the target velocity along the direction of the headin,
- and \\(\omega\\) is the target turn rate which represents the heading rate of change.

This motion model assumes that both the target velocity and turn rate are constant (\\(v\_{k+1} = v_k\\), \\(\omega\_{k+1} = \omega_k\\))

{{< rawhtml >}}
<image src="/posts/images/02_state.svg" alt="state and measurement model" position="center" style="border-radius: 8px; width: 550px; height: 320px; object-fit: cover; object-position: top;">
{{< /rawhtml >}}
Fig.1 Target state with CTRV and sensor measurement.

# EKF and UKF algorithm

![KF example GIF](/posts/images/KF_example.gif)

{{< rawhtml >}}

<iframe src="/posts/images/rmse_comparison.html" width=800 height=800 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

$$\lim_{x \to 0^+} \dfrac{1}{x} = \infty$$

# Conclusions

# Future steps and remaining questions
