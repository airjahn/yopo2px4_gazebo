#!/usr/bin/env python
# test4.py (YOPO + integrated PX4 takeoff & 5s hover)

import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand

from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

# MAVROS related
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")

# ===== MAVROS state (global, simple version) =====
current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


class YopoNet:
    def __init__(self, config, weight):
        self.config = config

        # --- logging ---
        self.log_counter = 0
        self.log_limit = 500
        log_file_name = "test4_log.txt"
        try:
            self.log_file = open(log_file_name, "w")
        except IOError:
            rospy.logerr("Could not open log file %s" % log_file_name)
            self.log_file = None

        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0

        # depth scale
        self.scale = {'435': 0.001, 'simulation': 1.0}.get(self.config.get('env', '435'), 1.0)
        if self.config.get('env', '435') != '435':
            rospy.logwarn("Depth encoding is 16UC1(mm) but env != '435'. Forcing scale=0.001.")
            self.scale = 0.001

        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config.get('plan_from_reference', True)
        if not self.plan_from_reference:
            rospy.logwarn("plan_from_reference is False; author recommended True for position control.")

        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler(
            'ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True
        ).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, map_location=self.device)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publishers
        self.lattice_traj_pub = rospy.Publisher(
            "/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1
        )
        self.best_traj_pub = rospy.Publisher(
            "/yopo_net/best_traj_visual", PointCloud2, queue_size=1
        )
        self.all_trajs_pub = rospy.Publisher(
            "/yopo_net/trajs_visual", PointCloud2, queue_size=1
        )
        self.ctrl_pub = rospy.Publisher(
            self.config["ctrl_topic"], PositionCommand, queue_size=1
        )

        # subscribers
        self.odom_sub = rospy.Subscriber(
            self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            self.config['depth_topic'], Image, self.callback_depth, queue_size=1
        )
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1
        )

        rospy.sleep(1.0)
        rospy.loginfo("YOPO Net Node initialized, waiting for start_planning_timer().")

    # ---------- logging ----------
    def log_to_file(self, message: str):
        if self.log_file is None or self.log_counter >= self.log_limit:
            return
        try:
            log_entry = "[Msg #{} at {:.4f}] {}\n".format(
                self.log_counter + 1, rospy.Time.now().to_sec(), message
            )
            self.log_file.write(log_entry)
            self.log_counter += 1
            if self.log_counter == self.log_limit:
                rospy.loginfo("Reached 500 messages, closing test4_log.txt.")
                self.log_file.close()
        except Exception as e:
            rospy.logerr("Failed to write to log file: %s" % e)
            if self.log_file:
                self.log_file.close()
            self.log_file = None

    # ---------- callbacks ----------
    def callback_set_goal(self, data: PoseStamped):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2.0])
        self.arrive = False
        goal_msg = "New Goal: ({:.1f}, {:.1f})".format(
            data.pose.position.x, data.pose.position.y
        )
        rospy.loginfo(goal_msg)
        self.log_to_file(goal_msg)

    def callback_odometry(self, data: Odometry):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array(
                (
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                )
            )
            self.desire_vel = np.array(
                (
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                )
            )
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat(
                [
                    self.odom.pose.pose.orientation.x,
                    self.odom.pose.pose.orientation.y,
                    self.odom.pose.pose.orientation.z,
                    self.odom.pose.pose.orientation.w,
                ]
            ).as_euler("ZYX", degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array(
            (
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            )
        )

        odom_log_msg = "ODOM - pos: {}, goal: {}, dist: ({})".format(
            pos, self.goal, np.linalg.norm(pos - self.goal)
        )
        rospy.loginfo(odom_log_msg)
        self.log_to_file(odom_log_msg)

        if np.linalg.norm(pos - self.goal) < 1.0 and not self.arrive:
            arrive_msg = "Arrive!"
            rospy.loginfo(arrive_msg)
            self.log_to_file(arrive_msg)
            self.arrive = True

    # ---------- process odom ----------
    def process_odom(self):
        Rotation_wb = R.from_quat(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        ).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        vel_w = (
            self.desire_vel
            if self.plan_from_reference
            else np.array(
                [
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                ]
            )
        )
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = np.array((0.0, 0.0, 0.0))
        acc_c = np.dot(Rotation_cw, acc_w)

        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm.to(self.device, non_blocking=True)

    # ---------- depth callback ----------
    @torch.inference_mode()
    def callback_depth(self, data: Image):
        if not self.odom_init:
            return

        # depth image process (assume 16UC1, mm)
        try:
            depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except Exception as e1:
            try:
                assert data.encoding == "16UC1", "Expected encoding '16UC1', got %s" % data.encoding
                depth = np.frombuffer(data.data, dtype=np.uint16).reshape(
                    data.height, data.width
                )
            except Exception as e2:
                err_msg = (
                    "\033[91mBoth cv_bridge and numpy fallback failed for 16UC1:\n"
                    "cv_bridge error: {}\n"
                    "numpy error: {}\033[0m"
                ).format(e1, e2)
                rospy.logerr(err_msg)
                self.log_to_file(err_msg)
                return

        depth = depth.astype(np.float32)

        time0 = time.time()
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )

        depth = np.minimum(depth * self.scale, self.max_dis) / self.max_dis

        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(
            np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS
        )
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])

        # YOPO inference
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_norm = self.process_odom()
        obs_input = self.state_transform.prepare_input(obs_norm)
        obs_input = obs_input.to(self.device, non_blocking=True)

        time2 = time.time()
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # Post-processing
        endstate, score = self.process_output(
            endstate_pred, score_pred, return_all_preds=self.visualize
        )
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(score_pred) if self.visualize else 0
        with self.lock:
            start_pos = (
                self.desire_pos
                if self.plan_from_reference
                else np.array(
                    (
                        self.odom.pose.pose.position.x,
                        self.odom.pose.pose.position.y,
                        self.odom.pose.pose.position.z,
                    )
                )
            )
            start_vel = (
                self.desire_vel
                if self.plan_from_reference
                else np.array(
                    (
                        self.odom.twist.twist.linear.x,
                        self.odom.twist.twist.linear.y,
                        self.odom.twist.twist.linear.z,
                    )
                )
            )
            self.optimal_poly_x = Poly5Solver(
                start_pos[0],
                start_vel[0],
                self.desire_acc[0],
                endstate_w[action_id, 0, 0] + start_pos[0],
                endstate_w[action_id, 0, 1],
                endstate_w[action_id, 0, 2],
                self.traj_time,
            )
            self.optimal_poly_y = Poly5Solver(
                start_pos[1],
                start_vel[1],
                self.desire_acc[1],
                endstate_w[action_id, 1, 0] + start_pos[1],
                endstate_w[action_id, 1, 1],
                endstate_w[action_id, 1, 2],
                self.traj_time,
            )
            self.optimal_poly_z = Poly5Solver(
                start_pos[2],
                start_vel[2],
                self.desire_acc[2],
                endstate_w[action_id, 2, 0] + start_pos[2],
                endstate_w[action_id, 2, 1],
                endstate_w[action_id, 2, 2],
                self.traj_time,
            )
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        if self.verbose:
            self.time_interpolation += time1 - time0
            self.time_prepare += time2 - time1
            self.time_forward += time3 - time2
            self.time_process += time4 - time3
            self.time_visualize += time5 - time4
            self.count += 1

            time_log_msg = (
                "TIME - depth-interpolation: {:.2f}ms; data-prepare: {:.2f}ms; "
                "network-inference: {:.2f}ms; post-process: {:.2f}ms; "
                "visualize-trajectory: {:.2f}ms"
            ).format(
                1000 * self.time_interpolation / self.count,
                1000 * self.time_prepare / self.count,
                1000 * self.time_forward / self.count,
                1000 * self.time_process / self.count,
                1000 * self.time_visualize / self.count,
            )
            rospy.loginfo(time_log_msg)
            self.log_to_file(time_log_msg)

    # ---------- control publisher ----------
    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = (
                self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            )
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(
                self.ctrl_time
            )
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(
                self.ctrl_time
            )
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(
                self.ctrl_time
            )

            self.desire_pos = np.array(
                [
                    control_msg.position.x,
                    control_msg.position.y,
                    control_msg.position.z,
                ]
            )
            self.desire_vel = np.array(
                [
                    control_msg.velocity.x,
                    control_msg.velocity.y,
                    control_msg.velocity.z,
                ]
            )
            self.desire_acc = np.array(
                [
                    control_msg.acceleration.x,
                    control_msg.acceleration.y,
                    control_msg.acceleration.z,
                ]
            )
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(
                self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt
            )
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

            pub_log_msg = (
                "PUBLISH - pos_cmd to {} with position x:{:.3f}, y:{:.3f}, z:{:.3f}"
            ).format(
                self.config["ctrl_topic"],
                control_msg.position.x,
                control_msg.position.y,
                control_msg.position.z,
            )
            self.log_to_file(pub_log_msg)

    # ---------- helper functions ----------
    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(
                endstate_pred[action_id, :][np.newaxis, :], lattice_id
            )
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(
                endstate_pred,
                torch.arange(self.lattice_primitive.traj_num - 1, -1, -1),
            )

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = (
            self.desire_pos
            if self.plan_from_reference
            else np.array(
                (
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                )
            )
        )
        start_vel = (
            self.desire_vel
            if self.plan_from_reference
            else np.array(
                (
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                )
            )
        )

        # best trajectory (single)
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack(
                (
                    self.optimal_poly_x.get_position(t_values),
                    self.optimal_poly_y.get_position(t_values),
                    self.optimal_poly_z.get_position(t_values),
                ),
                axis=-1,
            )
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)

        # lattice trajectories
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(
                start_pos[0],
                start_vel[0],
                self.desire_acc[0],
                lattice_endstate[:, 0] + start_pos[0],
                zero_state[:, 0],
                zero_state[:, 0],
                self.traj_time,
            )
            lattice_poly_y = Polys5Solver(
                start_pos[1],
                start_vel[1],
                self.desire_acc[1],
                lattice_endstate[:, 1] + start_pos[1],
                zero_state[:, 1],
                zero_state[:, 1],
                self.traj_time,
            )
            lattice_poly_z = Polys5Solver(
                start_pos[2],
                start_vel[2],
                self.desire_acc[2],
                lattice_endstate[:, 2] + start_pos[2],
                zero_state[:, 2],
                zero_state[:, 2],
                self.traj_time,
            )
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack(
                (
                    lattice_poly_x.get_position(t_values),
                    lattice_poly_y.get_position(t_values),
                    lattice_poly_z.get_position(t_values),
                ),
                axis=-1,
            )
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)

        # all trajectories with score as intensity
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(
                start_pos[0],
                start_vel[0],
                self.desire_acc[0],
                pred_endstate[:, 0, 0] + start_pos[0],
                pred_endstate[:, 0, 1],
                pred_endstate[:, 0, 2],
                self.traj_time,
            )
            all_poly_y = Polys5Solver(
                start_pos[1],
                start_vel[1],
                self.desire_acc[1],
                pred_endstate[:, 1, 0] + start_pos[1],
                pred_endstate[:, 1, 1],
                pred_endstate[:, 1, 2],
                self.traj_time,
            )
            all_poly_z = Polys5Solver(
                start_pos[2],
                start_vel[2],
                self.desire_acc[2],
                pred_endstate[:, 2, 0] + start_pos[2],
                pred_endstate[:, 2, 1],
                pred_endstate[:, 2, 2],
                self.traj_time,
            )
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack(
                (
                    all_poly_x.get_position(t_values),
                    all_poly_y.get_position(t_values),
                    all_poly_z.get_position(t_values),
                ),
                axis=-1,
            )
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def warm_up(self):
        depth = torch.zeros(
            (1, 1, self.height, self.width), dtype=torch.float32, device=self.device
        )
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)

    def start_planning_timer(self):
        rospy.loginfo("Starting YOPO planning timer...")
        self.timer_ctrl = rospy.Timer(
            rospy.Duration(self.ctrl_dt), self.control_pub
        )


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    p.add_argument("--trial", type=int, default=1, help="trial number")
    p.add_argument("--epoch", type=int, default=50, help="epoch number")
    return p


# ==============================
#  MAIN: takeoff + hover + YOPO
# ==============================
if __name__ == "__main__":
    # 1) init node
    rospy.init_node("yopo_net_with_takeoff", anonymous=False)

    # 2) MAVROS subscribers / publishers / services (from takeoff.py)
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
    local_pos_pub = rospy.Publisher(
        "mavros/setpoint_position/local", PoseStamped, queue_size=10
    )

    rospy.loginfo("Waiting for MAVROS services...")
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    rospy.loginfo("MAVROS services connected.")

    rate = rospy.Rate(20)
    TAKEOFF_ALTITUDE = 2.0
    HOVER_DURATION = rospy.Duration(5.0)

    # wait FCU connection
    rospy.loginfo("Waiting for FCU connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()
    rospy.loginfo("FCU connected.")

    hover_pose = PoseStamped()
    hover_pose.pose.position.x = 0.0
    hover_pose.pose.position.y = 0.0
    hover_pose.pose.position.z = TAKEOFF_ALTITUDE

    # send some initial setpoints
    rospy.loginfo("Sending initial setpoints...")
    for _ in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(hover_pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = "OFFBOARD"

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()
    takeoff_complete = False

    rospy.loginfo("Attempting to arm and switch to OFFBOARD...")
    while not rospy.is_shutdown() and not takeoff_complete:
        now = rospy.Time.now()
        if current_state.mode != "OFFBOARD" and (now - last_req) > rospy.Duration(5.0):
            if set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            last_req = now
        elif not current_state.armed and (now - last_req) > rospy.Duration(5.0):
            if arming_client.call(arm_cmd).success:
                rospy.loginfo("Vehicle armed")
            last_req = now

        if current_state.mode == "OFFBOARD" and current_state.armed:
            takeoff_complete = True
            rospy.loginfo("Takeoff procedure complete (armed + OFFBOARD).")

        local_pos_pub.publish(hover_pose)
        rate.sleep()

    # 5) hover 5s at 2m
    rospy.loginfo(
        "Hovering at {:.1f} m for {:.1f} s...".format(
            TAKEOFF_ALTITUDE, HOVER_DURATION.to_sec()
        )
    )
    hover_start = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - hover_start) < HOVER_DURATION:
        local_pos_pub.publish(hover_pose)
        rate.sleep()

    rospy.loginfo("Hover complete. Initializing YOPO planner...")

    # 6) init YOPO planner (original __main__ logic)
    args = parser().parse_args(rospy.myargv()[1:])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = (
        "yopo_trt.pth"
        if args.use_tensorrt
        else os.path.join(
            base_dir, "saved", "YOPO_{}".format(args.trial), "epoch{}.pth".format(args.epoch)
        )
    )
    print("load weight from:", weight)

    settings = {
        "use_tensorrt": args.use_tensorrt,
        "goal": [10, 0, 2],
        "env": "435",
        "pitch_angle_deg": -0.0,
        "odom_topic": "/mavros/local_position/odom",
        "depth_topic": "/camera/depth/image_raw",
        "ctrl_topic": "/yopo/pos_cmd",
        "plan_from_reference": False,
        "verbose": False,
        "visualize": True,
    }

    yopo_node = YopoNet(settings, weight)
    yopo_node.start_planning_timer()

    rospy.loginfo("YOPO planner is now active. Spinning.")
    rospy.spin()
