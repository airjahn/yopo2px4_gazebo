#!/usr/bin/env python3
# odom_publisher.py

import math
import threading
import numpy as np
import rospy
import tf2_ros

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from control_msg import PositionCommand
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge

class OdomShadowFromCmd:
    def __init__(self):
        # --- Frames, topics, rates ---
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/so3_control/pos_cmd")
        self.rate_hz = rospy.get_param("~rate_hz", 50)
        self.track_alpha = float(rospy.get_param("~track_alpha", 0.0)) # 0=snap

        # --- Initial pose (world) ---
        self.init_x = float(rospy.get_param("~init_x", 0.0))
        self.init_y = float(rospy.get_param("~init_y", 0.0))
        self.init_z = float(rospy.get_param("~init_z", 2.0)) # <-- start at z=2
        self.init_yaw = float(rospy.get_param("~init_yaw", 0.0))

        # --- Depth params ---
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.caminfo_topic = rospy.get_param("~caminfo_topic", "/camera/depth/camera_info")
        self.depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")
        self.img_width = int(rospy.get_param("~image_width", 640))
        self.img_height = int(rospy.get_param("~image_height", 480))
        self.depth_value_m = float(rospy.get_param("~depth_value_m", 10.0))
        self.hfov_deg = float(rospy.get_param("~hfov_deg", 70.0))

        # --- Pub/Sub & TF ---
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.tf_br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber(self.cmd_topic, PositionCommand, self._cmd_cb, queue_size=10)
        self.img_pub = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(self.caminfo_topic, CameraInfo, queue_size=1)
        self.bridge = CvBridge()

        # --- State ---
        self.lock = threading.Lock()
        self.last_cmd = None

        # Initialize published pose to (0,0,2), yaw=0
        self.x_pub, self.y_pub, self.z_pub, self.yaw_pub = self.init_x, self.init_y, self.init_z, self.init_yaw

        # Constant depth (meters)
        self.depth_img = np.full((self.img_height, self.img_width), self.depth_value_m, dtype=np.float32)
        self.cam_info = self._make_camera_info(self.img_width, self.img_height, self.hfov_deg, self.depth_frame)

    def _cmd_cb(self, msg: PositionCommand):
        with self.lock:
            self.last_cmd = msg

    @staticmethod
    def _make_camera_info(w, h, hfov_deg, frame_id):
        hfov = math.radians(hfov_deg)
        fx = fy = (w / 2.0) / math.tan(hfov / 2.0)
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        
        ci = CameraInfo()
        ci.header.frame_id = frame_id
        ci.width = w
        ci.height = h
        ci.distortion_model = "plumb_bob"
        ci.D = [0] * 5
        ci.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        ci.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        ci.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return ci

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                cmd = self.last_cmd

            # Defaults when no command yet
            x_ref, y_ref, z_ref = self.x_pub, self.y_pub, self.z_pub
            vx_ref, vy_ref, vz_ref = 0.0, 0.0, 0.0
            yaw_ref, yaw_rate = self.yaw_pub, 0.0

            # If we have a command, track it (optionally smoothed)
            if cmd is not None:
                x_ref, y_ref, z_ref = float(cmd.position.x), float(cmd.position.y), 2.0 #float(cmd.position.z)
                vx_ref, vy_ref, vz_ref = float(cmd.velocity.x), float(cmd.velocity.y), 0.0 #float(cmd.velocity.z)
                yaw_ref = float(getattr(cmd, "yaw", 0.0))
                yaw_rate = float(getattr(cmd, "yaw_dot", 0.0))

                a = self.track_alpha
                # position smoothing
                self.x_pub = (1.0 - a) * x_ref + a * self.x_pub
                self.y_pub = (1.0 - a) * y_ref + a * self.y_pub
                self.z_pub = (1.0 - a) * z_ref + a * self.z_pub
                # yaw smoothing with unwrap
                dyaw = (yaw_ref - self.yaw_pub + math.pi) % (2 * math.pi) - math.pi
                self.yaw_pub = self.yaw_pub + (1.0 - a) * dyaw
            # else: keep initial (0,0,2, yaw=0) and zero twists

            now = rospy.Time.now()
            quat = Quaternion(*quaternion_from_euler(0.0, 0.0, self.yaw_pub))

            # TF map->base_link
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x_pub
            t.transform.translation.y = self.y_pub
            t.transform.translation.z = self.z_pub
            t.transform.rotation = quat
            self.tf_br.sendTransform(t)

            # /odom
            od = Odometry()
            od.header.stamp = now
            od.header.frame_id = self.odom_frame
            od.child_frame_id = self.base_frame
            od.pose.pose.position.x = self.x_pub
            od.pose.pose.position.y = self.y_pub
            od.pose.pose.position.z = self.z_pub
            od.pose.pose.orientation = quat
            od.pose.covariance = [0.0] * 36
            od.twist.twist.linear.x = vx_ref
            od.twist.twist.linear.y = vy_ref
            od.twist.twist.linear.z = vz_ref
            od.twist.twist.angular.x = 0.0
            od.twist.twist.angular.y = 0.0
            od.twist.twist.angular.z = yaw_rate
            self.odom_pub.publish(od)

            # Depth image (constant)
            img_msg = self.bridge.cv2_to_imgmsg(self.depth_img, encoding="32FC1")
            img_msg.header.stamp = now
            img_msg.header.frame_id = self.depth_frame
            self.cam_info.header.stamp = now
            self.info_pub.publish(self.cam_info)
            self.img_pub.publish(img_msg)

            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("odom_shadow_from_pos_cmd_with_depth")
    OdomShadowFromCmd().spin()