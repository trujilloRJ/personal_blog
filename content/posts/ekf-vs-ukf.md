+++
title = "EKF vs. UKF performance and robustness comparison for nonlinear state estimation"
date = "2024-10-26T15:06:40+02:00"
author = "Javier Trujillo Rodriguez"
authorTwitter = "" #do not include @
cover = ""
tags = ["kalman-filters", "tracking", "numpy"]
keywords = ["kalman filters", "object tracking", "EKF", "UKF", "numpy"]
description = "A performance comparison between the Extended Kalman Filter (EKF) and the Unscented Kalman Filter (UKF) in a simulated environment"
showFullContent = false
readingTime = false
hideComments = false
math = true
+++

# Motivation

The Kalman Filter (KF) estimates the state of a dynamic system based on noisy measurements, delivering optimal performance if the **system is linear and noise is Gaussian**. However, in real-world nonlinear systems, these conditions are rarely met. To address this, the community had proposed variations of the original KF that are **sub-optimal** but still provide good performance. Among the proposed solutions, two approaches have been widely adopted, the Extended-KF (EKF) and the Unscented-KF (UKF).

The EKF approximates nonlinear functions by linearizing them around the current estimate. In practice, due to complexity and computational cost, only the first-order Taylor series expansion term is used for the linearization. In contrast, the UKF estimates the mean and covariance of the state distribution by selecting a minimal set of points (sigma points) from the current state, propagating these points through the nonlinear system, and using the resulting transformed points to calculate the posterior mean and covariance.

As engineers facing a decision between these two, we seek insight into the following questions:

- Which filter offers better overall performance?

- Which filter is computationally cheaper?

- Which filter is more robust to model mismatches?

This post explores each of these points by comparing the EKF and UKF in a simulated environment, assuming a target moving with Constant Turn Rate and Velocity (CTRV) and a sensor that measures range and bearing. Both the motion model and the measurement function are nonlinear, making this a perfect use case for comparing EKF and UKF.

## Source code

The source code for this post is avialable in the following Github repo:
https://github.com/trujilloRJ/kf_sandbox

## Table of Contents

1. [Motion model and measurement function](#motion-model-and-measurement-function)
2. [EKF and UKF algorithm](#ekf-and-ukf-algorithm)
3. [Implementation](#implementation)
4. [Simulation and validation metrics](#simulation-and-validation-metrics)
5. [Results](#results)
6. [Conclusions](#conclusions)
7. [Appendix](#appendix)

# Motion model and measurement function

#### CTRV motion model

We will consider a target moving on a 2D plane whose motion can be modeled with the CTRV. The filter state is represented in Fig. 1 and defined in polar coordinates as:

$$\bold{x_k}=[x_k, y_k, \phi_k, v_k, \omega_k]^T$$

Where the subscript \\(k\\) represents the time index and:

- \\(x\\) and \\(y\\) is the target position in the X and Y axis respectively,
- \\(\phi\\) is the target heading with respect to X axis,
- \\(v\\) is the target velocity along the direction of the heading,
- and \\(\omega\\) is the target turn rate which represents the heading rate of change.

{{< rawhtml >}}
<image src="/posts/images/02_state.svg" alt="state and measurement model" position="center" style="border-radius: 8px; width: 550px; height: 320px; object-fit: cover; object-position: top;">
{{< /rawhtml >}}
Fig.1 Target state with CTRV and sensor measurement.

This motion model assumes that both the target velocity and turn rate are constant, the state transition function can be obtained as:

{{< rawhtml >}}
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
{{< /rawhtml >}}

This function is clearly nonlinear, which means that if we assume the state distribution is Gaussian at time \\(k\\), passing it through this function will yield a distribution that is no longer Gaussian. This violates the assumptions of the standard KF, and using it here would lead to filter divergence. To address this, researchers have proposed various solutions, with the EKF being the most widely adopted in the industry. Implementing the EKF requires calculating Jacobians for the state transition function. For clarity and conciseness, these calculations are provided in the [Appendix](#appendix) section.

#### Range and bearing measurement function

As visible in Fig. 1, the sensor provides at each time \\(k\\) the target position by measuring range \\(r^m_k\\) and bearing \\(\theta^m_k\\) defining the measurement vector:

$$\bold{z_k}=[r^m_k, \theta^m_k]^T$$

However, the filter state contain the target position in cartesian coordinates. Therefore, to update it, we need a measurement function that maps between the state space and measurement space:

{{< rawhtml >}}
$$
\bold{\hat{z}_{k+1}}=h(\bold{\hat{x}})=
\begin{bmatrix}
\sqrt{x_k^2 + y_k^2} \\\
\tan^{-1}({y_k}/{x_k})
\end{bmatrix}
$$
{{< /rawhtml >}}

The measurement function is also nonlinear. The Jacobian of this function, required for the EKF, can also be found in the [Appendix](#appendix).

#### Modelling measurement and process noise

Due to noise, the sensor never provides an exact measurement. The noise is usually modeled as independent random Gaussian distributions with zero mean and covariance defined as:

{{< rawhtml >}}
$$
\bold{R}=
\begin{bmatrix}
\sigma^2_r & 0 \\\
0 & \sigma^2_{\theta}
\end{bmatrix}
$$
{{< /rawhtml >}}

Where \\(\sigma_r\\) and \\(\sigma\_{\theta}\\) represent the standard deviations of noise in range and bearing, respectively. These values are usually inherent to the sensor's characteristics. For instance, in FMCW radars, range error relates to range resolution, which in turn depends on the waveform bandwidth.

Additionally, we must consider inaccuracies in our motion model, as real targets rarely follow an exact CTRV pattern. Variations due to maneuvers and slight path deviations are common. To account for this, we introduce process noise as zero-mean Gaussian distributions applied to the target's velocity \\(q_v\\) and turn rate \\(q_{\omega}\\). The covariance matrix is then calculated as follows:

$$
\bold{Q_k}= \bold{\Gamma_k}\bold{q}\bold{\Gamma_k^T}
$$

where \\( \bold{q} = diag([q_v^2, q_\omega^2]) \\) and

{{< rawhtml >}}
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
{{< /rawhtml >}}

# EKF and UKF algorithm

#### EKF

As previously mentioned, the EKF handles system nonlinearities by linearizing them around the current estimate. In practice, the main difference from the standard KF is that the matrices used for covariance propagation are the Jacobians of the state transition and measurement functions. The EKF cycle is represented by the following equations:

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

An important note is that using the Jacobian, the EKF is aproximating the nonlinear function using only the first order term of the Taylor series expansion. Other, more accurate implementations of the EKF, incorporates [second order terms](https://www.diva-portal.org/smash/get/diva2:511843/FULLTEXT01.pdf) but its implementation is unfeasible for most practical systems.

#### UKF

The UKF tackles nonlinearity in a totally different way. Instead of trying to approximate the nonlinear system, it uses a method called the Unscented Transform (UT) to approximate the resulting probability distribution. This involves picking a set of special points, called sigma points, to represent the spread of possible states. These points are propagated through the nonlinear function, and from the results, the UKF estimates the new mean and covariance based on those transformed points.

Before diving into the UKF equations, it’s helpful to first understand the Unscented Transform. In this context, the UT can be defined as an operation that takes as input the state mean \\( \bold{x_k} \\), covariance \\( \bold{P_k} \\) and the nonlinear function \\( f \\). Then, the UT estimates the mean \\( \bold{y_m} \\) and covariance \\( \bold{P_y} \\) of the resulting distribution after passing the state through the nonlinear function. In addition, the propagated sigma points \\( \mathcal{Y} \\) are also of interest for UKF.

$$ (\bold{y_m},\bold{P_y}, \mathcal{Y}) = UT(f, \bold{x_k}, \bold{P_k}) $$

Step by step, the UT is doing the following:

1 Selecting parameters:

  $$\alpha = 1, \hspace{0.2cm} \beta = 2, \hspace{0.2cm} \kappa = 3 - n_x$$

  $$\lambda = \alpha^2(n_x+\kappa)-n_x$$

2 Computing weights

$$ w^m_0 = \frac{\lambda}{\lambda+n_x}, \hspace{0.5cm} w^c_0 = \frac{\lambda}{\lambda+n_x} + 1 - \alpha^2 + \beta $$

$$ w^m_i = w^c_i = \frac{1}{2(n_x+\lambda)} $$

3 Generating sigma points set \\( \mathcal{X} \\):

$$ \mathcal{X_0} = \bold{x_k} $$

$$ \mathcal{X_i} = \bold{x_k} + \left(\sqrt{(\lambda+n_x)P_k}\right)_i, \hspace{0.5cm} i=1,..,n_x $$

$$ \mathcal{X_i} = \bold{x_k} - \left(\sqrt{\lambda+n_x}\right)_{i-n_x}, \hspace{0.5cm} i=n_x + 1,..,2n_x $$

4 Propagating sigma points through nonlinear function (either  \\( f(.) \\) or \\( h(.) \\))

$$ \mathcal{Y_i} = f(\mathcal{X_i}), \hspace{0.5cm} i=0,...,2n_x$$

5 Estimating mean and covariance of the propagated distribution

$$ \bold{y_m} = \sum_{i=0}^{2n_x}w^m_i\mathcal{Y_i} $$

$$ \bold{P_y} = \sum_{i=0}^{2n_x}w^c_i(\mathcal{Y_i} - \bold{x}_m)(\mathcal{Y_i} - \bold{x}_m)^T $$

Where \\( n_x \\) is the dimension of the state and \\( \bold{L_k} = \sqrt{(\lambda+n_x)\bold{P_k}} \\) is the scaled square root of the state covariance matrix. The square root of a matrix is not defined, conventionally people use Cholesky factorization as an equivalent to matrix square root and that is what we used in the UKF implementation by leveraging `scipy.linalg.cholesky` function.

An excellent source to get intuition behind this and, in general, for KF is the [book by Roger Labbe](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/tree/master).

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

It is important to remark that the UKF does not require the computation of Jacobians as opposed to EKF. Despite this, it has been proven that UKF is able to achieve approximations of 3rd order for gaussian cases.

# Implementation

The implementation of both filters is available in `filters.py` module. The EKF under `EKF_CTRV` class and the UKF under `UKF_CTRV`. The motion models and measurement functions are designed as dependencies, allowing for greater flexibility if different motion models or measurement functions need to be tested in the future.

A note on the CTRV implementation: in the transition function, the model becomes undefined when the target’s turn rate is zero \\( \omega=0 \\). To handle these cases, the code simplifies both the transition function and its Jacobian to a Constant Velocity (CV) model.

#### Note on angle arithmetics

Angles are modular quantities and as such, additional checks need to be in place when dealing with them. For instance, we cannot compute directly the average of a set of angles: the mean angle between 359\\( \degree \\) and 1\\( \degree \\) is 0\\( \degree \\), not 180\\( \degree \\). Instead, we need to use the [circular mean](https://en.wikipedia.org/wiki/Circular_mean) to ensure accurate computations.

$$ \bar{\theta} = \tan^{-1}\left( \frac{\sum_{i=0}^{N-1} \sin(\theta_i)}{\sum_{i=0}^{N-1} \cos(\theta_i)} \right) $$

In our case, we have the bearing angle represented by \\( \theta \\) in the measurements and the heading angle represented by \\( \phi \\) in the state. It's important to ensure that (i) these angles always fall within the range of \\( (-\pi, \pi) \\), and (ii) we use the weighted circular mean when estimating the state mean in the UKF. The implementation for these is available in the `filters.py` module.

#### Note on UT parameter selection

The selection of UT parameters \\( \alpha, \beta \\) and \\( \kappa\\) research field of its own with proposals ranging from the [original paper by Julier](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/JulierUhlmann-UKF.pdf) to the [scaled singma points by Merwe](https://www.researchgate.net/publication/228846510_Sigma-Point_Kalman_Filters_for_Nonlinear_Estimation_and_Sensor-Fusion-Applications_to_Integrated_Navigation). Basically, these parameters controlled the spread of the sigma points around the mean which allow to approximate better (or not) the moments after nonlinear function.

However, another crucial aspect of parameter selection is its impact on numerical stability. For states with dimensions \\( n_x \geq 3 \\) the first weights for the mean and covariance \\( w^m_0 \\) and \\( w^c_0 \\) become large and negative. This could lead to covariance matrices that are no longer positive definite which causes filter divergence. Therefore, the selection we adopted here is to avoid this and guarantee UKF numerical 
stability. 

In fact, this topic has motivated an entire family of new nonlinear estimation filters. The [Cubature Kalman Filter (CKF)](https://ieeexplore.ieee.org/document/4982682) operates under the same principle as UKF but employs the cubature transformation to generate the sigma points, ensuring numerical stability.

# Simulation and validation metrics

A simulated target is generated to move in a straight line at a constant velocity, followed by a sharp turn, and then resumes straight-line motion. A sensor continuously provides noisy measurements of the target's range and bearing at each frame. Below is a table summarizing the parameters used for the simulation, including details about the sensor errors:

| Parameter        | Symbol | Value     |
| ---------------- | ------ | --------  |
| Time between frames | \\( T \\) | 200 ms     |
| Iterations | \\( N \\) | 10     |
| Sensor range standard deviation | \\( \sigma_r \\) | 0.5 m     |
| Sensor azimuth standard deviation | \\( \sigma_\theta \\) | 2\\( \degree \\)     |
| Process noise velocity | \\( q_v \\) | 1 m/s     |
| Process noise turn rate | \\( q_\omega \\) | 3\\( \degree \\)     |

All the simulation logic is contained in the `simulation.py` module. The figure below illustrates an example of one iteration. It’s important to note how the position errors in the X and Y coordinates increase as the target moves further away from the sensor. This behavior is expected, given the nonlinear nature of the conversion from polar to Cartesian coordinates.

{{< rawhtml >}}
<iframe src="/posts/images/example_simulation.html" width=800 height=600 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

The metric selected to evaluate the performance of the filters is the classical Root Mean Squared Error (RMSE) defined as:

$$ RMSE(\bold{x}) = \epsilon_x = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(\hat{x}_i-x_t)^2} $$

Where \\( N \\) is the number of iterations, \\( \hat{x}_i \\) is the filter estimation at the i-th iteration and \\( x_t \\) is the true value.

# Results

The entire code for the validation, including figures, can be found in the notebook `ekf_vs_ukf.ipynb`. The following figure shows the estimated state at each frame averaged by the number of iterations.

{{< rawhtml >}}
<iframe src="/posts/images/mean_state.html" width=800 height=600 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

As seen in the results, the EKF initially demonstrates faster convergence, particularly for velocity estimation. However, when the target begins its turning maneuver, both filters struggle to keep up with the true position and turn rate estimations. This lag is anticipated because the CTRV motion model does not account for changes in turn rate.

Notably, the UKF exhibits better performance during the turn; although its turn rate estimation is still somewhat delayed, it does not overshoot as significantly as the EKF. This characteristic results in a lower root mean square error (RMSE) for position estimation, as will be illustrated in the following figure.

As designers, we can enhance the performance of both filters by increasing the process noise \( q_w \). However, this adjustment comes with a trade-off, as it may lead to less accurate state estimations during periods when the target is moving in a straight line. Choosing appropriate values for the process noise is a topic worthy of its own post—perhaps something to explore in the future! 😉

To further illustrate our findings, the following figure displays the root mean square error (RMSE) over time for position (combining X and Y), heading, velocity, and turn rate.

{{< rawhtml >}}
<iframe src="/posts/images/rmse_comparison.html" width=800 height=800 allowTransparency="true" frameborder="0" scrolling="no"></iframe>
{{< /rawhtml >}}

First, let's focus on the position RMSE. In practice, this is the most important metric, since a big error in position could lead to misassociation and track loss which is critical for many applications. Overall, both filters offer better position estimation that what the raw measurements provide. Obviously, this is expected and one of the advantages of using a KF for state estimation. 

However, at the beginning of the target's turn, there is a brief period where both filters exhibit higher position RMSE than the measurements. This occurs due to filter lag caused by the target's maneuvers, which are not accounted for in the motion model. Among the two filters, the EKF performs the worst during this phase, as its RMSE peak is higher.

Additionally, the UKF outperforms the EKF in estimating the target's heading during the turning maneuver, consistently maintaining an RMSE below \( 10 \degree \). The same trend is observed for turn rate estimation, where the UKF shows superior performance. Conversely, the EKF demonstrates better estimation of target velocity during the maneuver compared to the UKF.

Just for the purpose of illustration, this GIF provides a zoomed view of the behaviours of both filters during target turning. The effect of lagging and later overshoot is visible for the EKF.

![KF example GIF](/posts/images/KF_example.gif)

#### Robustnes to process noise mismatch

For real applications, selecting the proper process noise is a challenging task. Filter designers must consider all potential maneuvers and movement patterns of the target. For instance, in automotive applications, a vehicle can brake, turn, cut in, and more. Additionally, motorcycles are likely to perform more maneuvers than cars. 

This selection process requires careful validation using extensive amounts of real, labeled data collected across various open road scenarios, as well as in simulated and controlled environments. Unfortunately, such data is often not readily available, leading to suboptimal process noise selections in many cases. Therefore, it is crucial to test the robustness of the filters against process noise mismatches to ensure reliable performance in real-world situations.

To perform this task, in the notebook `ekf_vs_ukf_robust_test.ipynb`, we have measured the RMSE of the filter estimations in `10` iterations with 4 different values for \\( q_v\\) and 5 for \\( q_\omega\\) for a total of 20 process noise combinations.

The following heatmaps show the averaged RMSE over iterations and time obtained for each filter using a given process noise combination. One key observation is that \\( q_\omega \\) has a more significant impact on performance than \\( q_v \\). Notably, both filters demonstrate their worst performance when \\( q_\omega = 0.5 \degree \\). This is expected, as this value fails to adequately capture the maneuvers performed by the target. Despite this, the UKF maintains significantly better position estimation than the EKF under these extreme conditions.

{{< rawhtml >}}
<image src="/posts/images/rmse_pos.png" alt="state and measurement model" position="center" style="border-radius: 8px; width: 800px;">
{{< /rawhtml >}}

{{< rawhtml >}}
<image src="/posts/images/rmse_phi.png" alt="state and measurement model" position="center" style="border-radius: 8px; width: 800px; height: 320px; object-fit: cover; object-position: top;">
{{< /rawhtml >}}

In general, the UKF seems to be more robust to mismatches in process noise, starting from \\( q_\omega= 0.8\degree\\) the position RMSE is reasonable and much lower than EKF for the same value of process noise. This conclusion is also valid for the heading estimation. 

However, the EKF can achieve comparable, and in some cases slightly better, performance than the UKF when the process noise values are appropriately chosen. This improved performance, though, comes at the expense of greater sensitivity to mismatches. Consequently, if the EKF is the chosen filter, designers must devote considerable attention and resources to selecting the process noise carefully to ensure optimal performance. 

#### Computational cost

The computational cost was measured in terms of time of execution per frame for each filter. A simple validation was conducted using `time.perf_counter_ns` from Python `time` module on each iteration and frame and later on reporting the average time per execution for each filter.

| Filter        | Average time per frame (ms)
| ---------------- | ------
| EKF | 100 us
| UKF | 620 us 

The reported values are not particularly relevant in absolute terms, as they can vary significantly based on the hardware platform and the programming language used for implementation. However, in relative terms, the results are noteworthy. Specifically, the EKF is approximately `6x` faster than the UKF when using the CTRV model with range and bearing measurement functions. This difference is especially important in scenarios where the implementation is aimed at embedded environments with limited computational resources.

As a word of caution, I cannot guarantee that my implementation is the fastest for UKF and EKF. Optimizing for speed was not the primary focus of this experiment. However, I made an effort to vectorize as many operations as possible by leveraging the functionality of the `numpy` library.

# Conclusions

The UKF and EKF were implemented using CTRV motion model and tested in a simulated environment with a sensor providing range and bearing. The results lead to the following conclusions:

- When desing properly, **both EKF and UKF are effective solutions for nonlinear filtering, offering similar performance**.

- **The UKF is more robust to process noise mismatch than the EKF**, making it preferable for applications characterized by a high degree of uncertainty where robustness is critical.

- **The EKF is computationally cheaper than UKF**, which can be a significant advantage for applications that need to run in resource-constrained environments.

- **A hidden advantage of the UKF over the EKF is that it does not require the computation of Jacobians**. While this is not a major concern for this experiment—since both the measurement and transition functions are analytically derivable—this could be a significant advantage in more complex scenarios where Jacobians are difficult or impractical to compute.

# Future steps and remaining questions

It is important to remark that this evaluation and comparison is by no means exhaustive. A logical next step would be to test additional scenarios, such as what happens when the velocity changes. For instance, how would the filters perform if we introduce tangential acceleration to the target?

Also, these conclusions cannot extrapolated completely to other applications. The performance of each filter is influenced by the specific nature of the nonlinear functions involved. Other factors, such as the time between frames and the intensity of measurement noise, also contribute to variations in filter performance, leading to different outcomes. 

A natural next step will be to incorporate Doppler velocity into the measurement function, since radar sensor are able to measure this value. This addition would make the measurement function even more nonlinear, potentially altering the performance of the filters. Exploring these variations will provide deeper insights into the capabilities and limitations of both the EKF and UKF in diverse situations.

On the same idea, it could be interesting to extend the motion model to Constant Turn Rate and Acceleration (CTRA). This more complex motion model can simultaneously capture both acceleration and turning maneuvers, providing a more realistic representation of target behavior in dynamic environments. By implementing the CTRA model, we could assess how well the EKF and UKF adapt to more intricate scenarios, potentially revealing additional strengths and weaknesses of each filter.

Please contact me via Github if you have any suggestions or want to see any of these steps developed 😊

# Appendix

1- Jacobian of the transition function:

{{< rawhtml >}}
$$
\bold{\dot{F}_k}=
\begin{bmatrix}
1 & 0 & r_c\left(\cos(\phi\_{k+1|k}) - \cos(\phi_k)\right) & 1/\omega_k \left(\sin(\phi\_{k+1|k}) - \sin(\phi_k)\right) & w_0 
\\\
0 & 1 & r_c\left(\sin(\phi\_{k+1|k}) - \sin(\phi_k)\right) & 1/\omega_k \left(-\cos(\phi\_{k+1|k}) + \cos(\phi_k)\right) & w_1
\\\
0 & 0 & 1 & 0 & T 
\\\
0 & 0 & 0 & 1 & 0 
\\\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$
{{< /rawhtml >}}

$$r_c = v_k/\omega_k$$

$$\phi_{k+1|k} = \phi_k + \omega_kT$$

$$ w_0 = r_cT\cos(\phi\_{k+1|k}) + r_c/\omega_k\left(\sin(\phi_k) - \sin(\phi\_{k+1|k})\right) $$

$$ w_1 = r_cT\sin(\phi\_{k+1|k}) - r_c/\omega_k\left(\cos(\phi_k) - \cos(\phi\_{k+1|k})\right) $$

2- Jacobian of the measurement function:

{{< rawhtml >}}
$$
\bold{\dot{H}_k}=
\begin{bmatrix}
x_k/r_k & y_k/r_k & 0 & 0 & 0 
\\\
-y_k/r_k^2 & x_k/r_k^2 & 0 & 0 & 0 
\\\
\end{bmatrix}
$$
{{< /rawhtml >}}

$$ r_k = \sqrt{x_k^2 + y_k^2} $$

