#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev
#
# Robert Twomey | 2025 | rtwomey@ucsd.edu | cohab-lab.net

"""Universal Robots lite6 arm tracking a moving target with self collision barriers."""

import argparse

import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter

import pinocchio as pin
import os

import meshcat_shapes
import pink
from pink import solve_ik
from pink.barriers import PositionBarrier
from pink.barriers import SelfCollisionBarrier
from pink.utils import process_collision_pairs

from pink.barriers import BodySphericalBarrier

from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        help="print out task errors and CBF values during execution",
        default=False,
        action="store_true",
    )
    
    # model, visual_model, collision_model = pin.Model(), pin.GeometryModel(), pin.GeometryModel()


    args = parser.parse_args()
    
    # robot = load_robot_description("xarm7_mj_description", root_joint=None)
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "../robots",
        "lite6.urdf",
    )

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {robot}")

    viz = start_meshcat_visualizer(robot)

    end_effector_task = FrameTask(
        "link_eef",
        position_cost=30.0, #10.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    # starting point (joint angles)
    q_ref = np.array(
        # [
        #     1.27153374,
        #     -0.87988708,
        #     1.89104795,
        #     1.73996951,
        #     -0.24610945,
        #     -0.74979019,
        # ]
        [ 0, 0, 0, 0, 0, 0 ]
    )

    # pos_barrier = PositionBarrier(
    #     "link_eef",
    #     indices=[1],
    #     p_min=np.array([-0.4]),
    #     p_max=np.array([0.6]),
    #     gain=np.array([100.0]),
    #     safe_displacement_gain=1.0,
    # )
    # barriers = [pos_barrier]


    # ee_frame_placement = pin.SE3()
    # ee_frame_placement.translation = np.array([0.0, 0.0, 0.05])
    # ee_frame_placement.rotation = np.eye(3)

    # ee_frame = pin.Frame(
    #     "lite6_ee_barrier",
    #     robot.model.getJointId("joint_eef"),
    #     robot.model.getFrameId("link_eef"),
    #     ee_frame_placement,
    #     pin.FrameType.OP_FRAME,
    # )

    # robot.model.addFrame(ee_frame)

    # robot.data = pin.Data(robot.model)

    # # Pink barriers
    # ee_barrier = BodySphericalBarrier(
    #     ("lite6_ee_barrier"),
    #     d_min=0.25,
    #     gain=100.0,
    #     safe_displacement_gain=1.0,
    # )
    # barriers = [ee_barrier]

    srdf_path = os.path.join(
        os.path.dirname(__file__),
        "../robots",
        "lite6.srdf",
    )
    print(srdf_path)

    # Collisions: processing collisions from urdf (include all) and srdf (exclude specified)
    # and updating collision model and creating corresponding collision data
    robot.collision_data = process_collision_pairs(robot.model, robot.collision_model, srdf_path)

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,  # Collision model is required for self_collision_barrier
        collision_data=robot.collision_data,
    )
    # print(robot.collision_model.collisionPairs)

    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=20.0,
        safe_displacement_gain=1.0,
        d_min=0.01,
    )
    barriers = [collision_barrier]

    tasks = [end_effector_task, posture_task]

    configuration = pink.Configuration(robot.model, robot.data, q_ref, collision_model=robot.collision_model)
    
    # configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        # end_effector_target.translation[0] = 0.3
        # end_effector_target.translation[1] = 0.5 * np.sin(t/4)
        # end_effector_target.translation[2] = 0.5 + 0.2 * np.sin(t/8)
        
        mu = 0.5
        x = 0.5 * np.cos(t*mu) #0.4
        y = 0.5 * np.sin((t*mu))
        z = 0.5+0.3 * np.cos((t*mu))
        # z = 0.5 + 0.2 * np.sin(t)
        # y = 0.5 * np.sin(t / 4)
        # z = 0.5 + 0.2 * np.sin(t / 8)

        # Set position
        end_effector_target.translation[0] = x
        end_effector_target.translation[1] = y
        end_effector_target.translation[2] = z

        # Compute direction from target to origin
        target_pos = np.array([x, y, z])
        # look_dir = -target_pos / np.linalg.norm(target_pos)  # normalized vector toward origin
        look_dir = target_pos / np.linalg.norm(target_pos) # away from origin

        # Choose an arbitrary "up" vector (e.g., world z-axis)
        up = np.array([0, 0, 1])

        # Recompute a right-handed frame (rotation matrix)
        right = np.cross(up, look_dir)
        right /= np.linalg.norm(right)
        new_up = np.cross(look_dir, right)

        # Build rotation matrix
        R = np.column_stack((right, new_up, look_dir))  # 3x3 rotation matrix

        # Set rotation (assuming pink uses something like SE(3))
        end_effector_target.rotation = R  # or maybe .linear if it's a Transform3d

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)

        # G, h = collision_barrier.compute_qp_inequalities(configuration, dt=dt)
        # # G, h = pos_barrier.compute_qp_inequalities(configuration, dt=dt)
        # distance_to_manipulator = configuration.get_transform_frame_to_world(
        #     "link_eef"
        # ).translation[1]
        # if args.verbose:
        #     print(
        #         f"Task error: {end_effector_task.compute_error(configuration)}"
        #     )
        #     print(
        #         "Position CBF value: "
        #         f"{pos_barrier.compute_barrier(configuration)[0]:0.3f} >= 0"
        #     )
        #     print(f"Distance to manipulator: {distance_to_manipulator} <= 0.6")
        #     print("-" * 60)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt