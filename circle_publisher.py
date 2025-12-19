#!/usr/bin/env python3
# circle_publisher.py

import math
import numpy as np
import rospy
import tf2_ros

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge

def make_camera_info(width, height, hfov_deg, frame_id):
    # Simple pinhole intrinsics from FOV
    hfov = math.radians(hfov_deg)
    fx = fy = (width / 2.0) / math.tan(hfov / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    ci = CameraInfo()
    ci.header.frame_id = frame_id
    ci.width = width
    ci.height = height
    ci.distortion_model = "plumb_bob"
    ci.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    ci.K = [fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0]
    ci.R = [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]
    ci.P = [fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0]
    return ci

def main():
    rospy.init_node("fixed_odom_and_depth_publisher")

    # Params
    rate_hz = rospy.get_param("~rate_hz", 30)
    yaw_rate = rospy.get_param("~yaw_rate_rad_s", 1.0)
    radius = rospy.get_param("~circle_radius_m", 3.0)       # meters
    omega = rospy.get_param("~angular_speed_rad_s", 0.5)    # rad/s
    cx = rospy.get_param("~center_x", 0.0)
    cy = rospy.get_param("~center_y", 0.0)
    z0 = rospy.get_param("~z_height_m", 2.0)
    odom_frame = rospy.get_param("~odom_frame", "world")
    base_frame = rospy.get_param("~base_frame", "base_link")
    depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")
    depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
    caminfo_topic = rospy.get_param("~caminfo_topic", "/camera/depth/camera_info")
    odom_topic = rospy.get_param("~odom_topic", "/odom")

    img_width = int(rospy.get_param("~image_width", 640))
    img_height = int(rospy.get_param("~image_height", 480))
    depth_value_m = float(rospy.get_param("~depth_value_m", 10.0))
    void_radius_px = int(rospy.get_param("~void_radius_px", min(img_width, img_height)//8))
    hfov_deg = float(rospy.get_param("~hfov_deg", 70.0))

    # Publishers
    odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)
    img_pub = rospy.Publisher(depth_topic, Image, queue_size=10)
    info_pub = rospy.Publisher(caminfo_topic, CameraInfo, queue_size=10)

    # TF broadcaster
    tf_br = tf2_ros.TransformBroadcaster()

    # Camera info (static except header stamp)
    cam_info = make_camera_info(img_width, img_height, hfov_deg, depth_frame)

    # Prebuild a base depth image with a circular NaN void
    depth = np.full((img_height, img_width), depth_value_m, dtype=np.float32)

    bridge = CvBridge()
    rate = rospy.Rate(rate_hz)

    t0 = rospy.Time.now()

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        t = (now - t0).to_sec()

        # Circle position
        x = cx + radius * math.cos(omega * t)
        y = cy + radius * math.sin(omega * t)
        z = z0

        # Velocities (world frame)
        vx = -radius * omega * math.sin(omega * t)
        vy = radius * omega * math.cos(omega * t)
        vz = 0.0

        # yaw = (yaw_rate * t) % (2.0 * math.pi)  # spinning aroud z
        # Yaw aligned with direction of travel (tangent)
        yaw = math.atan2(vy, vx)       # faces along velocity
        q = quaternion_from_euler(0.0, 0.0, yaw)
        quat = Quaternion(*q)

        # ===== Publish TF map -> base_link =====
        tfs = TransformStamped()
        tfs.header.stamp = now
        tfs.header.frame_id = odom_frame
        tfs.child_frame_id = base_frame
        tfs.transform.translation.x = x
        tfs.transform.translation.y = y
        tfs.transform.translation.z = z
        tfs.transform.rotation = quat
        tf_br.sendTransform(tfs)

        # ===== Publish /odom =====
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = odom_frame
        odom.child_frame_id = base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation = quat
        odom.pose.covariance = [0.0]*36

        # Linear velocity in world (typical for /odom)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz

        # Angular velocity (yaw rate) about +Z
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = omega

        odom_pub.publish(odom)

        # ===== Publish depth image & camera info =====
        img_msg = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        img_msg.header.stamp = now
        img_msg.header.frame_id = depth_frame
        cam_info.header.stamp = now
        info_pub.publish(cam_info)
        img_pub.publish(img_msg)

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass