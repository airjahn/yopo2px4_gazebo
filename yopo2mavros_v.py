#!/usr/bin/env python3
# yopo_to_mavros_bridge.py (Velocity Control Version)

import rospy
# !! 修改导入 !!
# from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped # 导入 TwistStamped
from tf.transformations import quaternion_from_euler # 这个可能不再需要，除非你想额外处理yaw

# 导入 YOPO 使用的自定义消息类型
from control_msg import PositionCommand # 这个保持不变

class YopoBridge:
    def __init__(self):
        rospy.init_node('yopo_to_mavros_bridge_velocity', anonymous=True) # 可以改个节点名

        # !! 修改发布者 !!
        # self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10) # 修改话题名和类型

        # 订阅者保持不变
        self.yopo_sub = rospy.Subscriber('/yopo/pos_cmd', PositionCommand, self.yopo_cmd_callback) # 或 /so3_control/pos_cmd 取决于你的test5.py设置

        rospy.loginfo("YOPO to MAVROS velocity bridge is running.")

    def yopo_cmd_callback(self, msg):
        """
        当收到来自 YOPO 的 PositionCommand 消息时，此函数被调用。
        它会将消息中的速度和偏航角速度转换为 TwistStamped 并发布。
        """
        # --- (日志记录部分可以保留或修改) ---
        rospy.loginfo_throttle(1.0,"=========================================================") # 使用 throttle 避免刷屏
        rospy.loginfo_throttle(1.0,"RECEIVED Original PositionCommand from YOPO:")
        rospy.loginfo_throttle(1.0, str(msg.velocity)) # 完整打印可能刷屏太快

        # !! 修改消息创建逻辑 !!
        # 创建 TwistStamped 消息
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        # frame_id 通常设为 'map' 或 'odom' 表示世界坐标系下的速度
        # 如果 YOPO 输出的速度是在机体坐标系下，则设为 'base_link' 或 'odom'（需确认）
        # 假设 YOPO 输出的是世界坐标系下的速度
        twist_msg.header.frame_id = "world"

        # 从 PositionCommand 填充 TwistStamped 的 twist 字段
        twist_msg.twist.linear.x = msg.velocity.x
        twist_msg.twist.linear.y = msg.velocity.y
        twist_msg.twist.linear.z = 0  # 不需要垂直速度

        # 角速度通常只用 z 轴 (yaw rate)
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = msg.yaw_dot

        # --- (日志记录部分可以保留或修改) ---
        rospy.loginfo_throttle(1.0,"----------------- CONVERTED TO -----------------")
        rospy.loginfo_throttle(1.0,"SENDING Converted TwistStamped to /mavros/setpoint_velocity/cmd_vel:")
        rospy.loginfo_throttle(1.0, f"Linear Vel: [x:{twist_msg.twist.linear.x:.2f}, y:{twist_msg.twist.linear.y:.2f}, z:{twist_msg.twist.linear.z:.2f}], Angular Vel z: {twist_msg.twist.angular.z:.2f}")
        rospy.loginfo_throttle(1.0,"=========================================================\n")

        # 发布转换后的消息给 MAVROS
        self.vel_pub.publish(twist_msg)

if __name__ == '__main__':
    try:
        # 确保 control_msg.py 能被找到 (如果需要)
        import sys
        import os
        # sys.path.append(os.path.dirname(__file__)) # 如果 control_msg 在同目录或 PYTHONPATH

        bridge = YopoBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass