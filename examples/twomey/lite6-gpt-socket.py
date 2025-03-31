#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Universal Robots lite6 arm tracking a moving target with self collision barriers and streaming joint data over socket."""

import argparse
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
import pinocchio as pin
import os
import socket
import sys
import meshcat_shapes
import pink
from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.utils import process_collision_pairs
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, try `[conda|pip] install robot_descriptions`"
    ) from exc

# Socket streaming config
streamJoints = True
connected = False
txsocket = socket.socket()
txsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
txport = 12346
txsocket.bind(('', txport))
txsocket.listen(5)
txconn = None

# Function to wait for a connection
def wait_for_connection():
    global txconn, connected
    print("tx: waiting for connection...")
    txconn, txaddr = txsocket.accept()
    connected = True
    print("tx: accepted connection from", str(txaddr[0]), ":", str(txaddr[1]))
    txconn.send(b"ready")
    print("waiting to move to initial position...", end="")
    sys.stdout.flush()
    data = txconn.recv(1024)
    print("done! going", data.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", help="print task errors and CBF values", default=False, action="store_true")
    args = parser.parse_args()

    urdf_path = os.path.join(os.path.dirname(__file__), "../robots", "lite6.urdf")

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {robot}")

    viz = start_meshcat_visualizer(robot)

    end_effector_task = FrameTask("link_eef", position_cost=30.0, orientation_cost=1.0)
    posture_task = PostureTask(cost=1e-3)

    q_ref = np.array([0, 0, 0, 0, 0, 0])

    srdf_path = os.path.join(os.path.dirname(__file__), "../robots", "lite6.srdf")
    robot.collision_data = process_collision_pairs(robot.model, robot.collision_model, srdf_path)

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        collision_data=robot.collision_data,
    )

    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=20.0,
        safe_displacement_gain=1.0,
        d_min=0.01,
    )
    barriers = [collision_barrier]
    tasks = [end_effector_task, posture_task]

    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    solver = "osqp" if "osqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0

    if streamJoints:
        wait_for_connection()

    while True:
        mu = 0.5
        x = 0.5 * np.cos(t * mu)
        y = 0.5 * np.sin(t * mu)
        z = 0.5 + 0.3 * np.cos(t * mu)

        target_pos = np.array([x, y, z])
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[:] = target_pos

        look_dir = target_pos / np.linalg.norm(target_pos)
        up = np.array([0, 0, 1])
        right = np.cross(up, look_dir)
        right /= np.linalg.norm(right)
        new_up = np.cross(look_dir, right)
        R = np.column_stack((right, new_up, look_dir))
        end_effector_target.rotation = R

        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(configuration.get_transform_frame_to_world(end_effector_task.frame).np)

        velocity = solve_ik(configuration, tasks, dt, solver=solver, barriers=barriers, safety_break=False)
        configuration.integrate_inplace(velocity, dt)

        if streamJoints:
            if connected:
                try:
                    txconn.send(str(configuration.q.tolist()).encode())
                except:
                    connected = False
                    print("tx: disconnected.")
            else:
                wait_for_connection()

        viz.display(configuration.q)
        rate.sleep()
        t += dt
