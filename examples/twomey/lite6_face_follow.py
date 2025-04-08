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

import socket
import sys
import json  # Ensure JSON is imported
import ast  # Add import for safely evaluating received data

streamJoints = True  # Toggle for streaming
q = None

if streamJoints:
    txport = 12346       # Port for joint streaming

    # Create socket outside main loop
    txsocket = socket.socket()
    txsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    txsocket.bind(('', txport))
    txsocket.listen(5)

    connected = False
    txconn = None

if streamJoints:
    rxport = 12347  # Port for receiving face coordinates
    rxsocket = socket.socket()
    rxsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rxsocket.bind(('', rxport))
    rxsocket.listen(5)

    rxconnected = False
    rxconn = None

# Wait for connection before starting simulation
def wait_for_connection():
    global txconn, connected, q
    while not connected:
        print("tx: waiting for connection...")
        txconn, txaddr = txsocket.accept()
        connected = True
    
    print("tx: accepted connection from", str(txaddr[0]), ":", str(txaddr[1]))
    txconn.send(json.dumps(q).encode())  # Send initial joint data as JSON
    
    # wait for reply
    print("waiting to move to initial position...", end="")
    sys.stdout.flush()
    data = txconn.recv(1024)
    print("done! going", data)

# Wait for connection to receive face data
def wait_for_rx_connection():
    global rxconn, rxconnected
    while not rxconnected:
        print("rx: waiting for connection...")
        rxconn, rxaddr = rxsocket.accept()
        rxconnected = True
        print("rx: accepted connection from", str(rxaddr[0]), ":", str(rxaddr[1]))

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
        [ 0, 0, 0, 0, 0, 0 ]
    )

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

    face_direction = None  # Initialize face direction

    end_effector_target = end_effector_task.transform_target_to_world
    
    # Set position
    end_effector_target.translation[0] = 0.3
    end_effector_target.translation[1] = 0.0#0.4
    end_effector_target.translation[2] = 0.2

    while True:
        # Receive face coordinates
        if not rxconnected:
            wait_for_rx_connection()


        try:
            data = rxconn.recv(1024)
            if data == b'':
                print("rx: socket closed.")
                rxconnected = False
                break

            if data:
                message = data.decode()
                try:
                    # Parse the received JSON message
                    parsed_data = json.loads(message)
                    thistype = parsed_data.get("type")
                    
                    thispayload = parsed_data.get("payload")
                except json.JSONDecodeError:
                    print("rx: problem decoding JSON data", message)
                    continue

                if thistype not in ["face", "relax", "rand", "pos", "reset", "wrist"]:
                    print("rx: unknown command", thistype)
                    continue

                if thistype == "face":
                    y, p, r, dx, dy, dz = thispayload
                    weight = 0.1#0.3
                    dx = weight * dx
                    dy = weight * dy
                    dz = weight * dz
                    face_direction = [dx, dy, dz]
                else:
                    face_direction = None

        except Exception as e:
            print("rx: error receiving data:", e)
            rxconnected = False

        # Update end_effector_target based on face_direction
        # face_direction = None
        if face_direction:
            dx, dy, dz = face_direction
            end_effector_target.translation[0] += dx
            end_effector_target.translation[1] += dy
            end_effector_target.translation[2] += dz

        # Compute direction from target to origin
        target_pos = np.array(end_effector_target.translation)
        look_dir = target_pos / np.linalg.norm(target_pos)  # Away from origin

        # Choose an arbitrary "up" vector (e.g., world z-axis)
        up = np.array([0, 0, 1])

        # Recompute a right-handed frame (rotation matrix)
        right = np.cross(up, look_dir)
        right /= np.linalg.norm(right)
        new_up = np.cross(look_dir, right)

        # Build rotation matrix
        R = np.column_stack((right, new_up, look_dir))  # 3x3 rotation matrix

        # Set rotation
        end_effector_target.rotation = R

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
        q = configuration.q.tolist()

        # Stream joint values over socket
        if streamJoints:
            if connected:
                try:
                    txconn.send(json.dumps(configuration.q.tolist()).encode())  # Send joint data as JSON
                except:
                    connected = False
                    print("tx: disconnected.")
            else:
                wait_for_connection()

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt