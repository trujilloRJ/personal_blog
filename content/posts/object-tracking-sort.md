+++
title = "Camera-based object tracking: An implementation of SORT algorithm"
date = "2025-02-28T15:06:40+02:00"
author = "Javier Trujillo Rodriguez"
authorTwitter = "" #do not include @
cover = ""
tags = ["computer-vision", "kalman-filters", "tracking", "opencv"]
keywords = ["kalman filters", "object tracking", "numpy"]
description = "This blog explores camera-based object tracking through the implementation of the Simple Online and Realtime Tracker (SORT) algorithm. It covers key concepts and foundational elements essential for object tracking applications, demonstrating them with a SORT implementation for automotive tracking using the KITTI dataset. While SORT is computationally efficient and relatively simple, it delivers solid performance in the KITTI object tracking challenge."
showFullContent = false
readingTime = false
hideComments = false
math = true
+++

{{< rawhtml >}}
<image src="/posts/images/introduction_sample.PNG" alt="Example lane" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}

The image above depicts a typical automotive scene captured by a monocular camera at a specific time. The colored boxes represent object detections made by a CNN-based detector, such as YOLO, with each color indicating a different class—green for cars and dark blue for pedestrians. However, these detections are independent across frames since the object detector lacks memory, meaning it does not recognize that an object in one frame is the same in the next.

In automotive applications, detecting objects in isolation is not sufficient. To enable functions like collision avoidance, path planning, and autonomous navigation, it is crucial to consistently identify and track objects as they move through consecutive frames. This process, known as object tracking, assigns a unique identity to each detected object, ensuring continuity and a more comprehensive understanding of the dynamic environment. 

In this post, we will explore key concepts of object tracking and present its flow diagram to illustrate the process. Additionally, we will implement a simple yet effective object tracker, widely recognized in the community as the Simple Online and Realtime Tracker (SORT), demonstrating its practical applications to the KITTI object tracking dataset.

# Introduction

Object tracking is the task of consistently identifying and following objects as they move across consecutive frames in a video. Given an initial detection of objects in an image sequence, the goal is to assign a unique identity to each object and maintain its trajectory over time, despite challenges such as occlusions, appearance changes, and complex motion patterns.
An effective tracking algorithm must address:
Object Association: Accurately matching objects across frames while minimizing identity switches.
Robustness to Occlusion: Handling partial or complete occlusions and re-identifying objects when they reappear.
Computational Efficiency: Operating in real-time with limited processing power, especially in resource-constrained applications.
Scalability: Adapting to multiple objects in dynamic environments without a significant drop in performance.
This problem is critical for various applications, including autonomous driving, surveillance, robotics, and augmented reality, where real-time and accurate object tracking is essential for decision-making and interaction with the environment.

Object tracking aims to solve the problem of consistently identifying and following objects as they move through a sequence of images or video frames. The key challenges it addresses include:
2. Object Association Across Frames – Ensuring that detected objects in consecutive frames are correctly matched, even when they undergo changes in position, appearance, or partial occlusion.
3. Handling Occlusions and Disruptions – Objects may be temporarily hidden behind other objects or leave and re-enter the frame, requiring the algorithm to maintain tracking consistency.
4. Real-Time Performance – Many applications, such as autonomous driving and surveillance, require object tracking to operate in real time with minimal computational cost.
Dealing with Appearance Variability – Objects can change in size, shape, orientation, and lighting conditions, making it challenging to track them reliably.
By solving these problems, object tracking enables applications like autonomous navigation, activity recognition, traffic monitoring, and augmented reality.


{{< rawhtml >}}
<image src="/posts/images/SORT_diagram.png" alt="Tracking diagram" position="center" style="border-radius: 8px; width: 1200px" >
{{< /rawhtml >}}
Fig.1 Core tracking system diagram

## Source code

## Table of Contents



