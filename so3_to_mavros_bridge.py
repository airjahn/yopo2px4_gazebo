#!/usr/bin/env python3
# yopo_to_mavros_bridge.py (Clean Version)

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

# 导入 YOPO 使用的自定义消息类型
from control_msg import PositionCommand

class YopoBridge:
    def __init__(self):
        rospy.init_node('yopo_to_mavros_bridge', anonymous=True)

        # 1. MAVROS 控制指令的发布者
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        # 2. YOPO 控制指令的订阅者
        self.yopo_sub = rospy.Subscriber('/yopo/pos_cmd', PositionCommand, self.yopo_cmd_callback)
# ---     用于文件日志记录 ---
        self.log_counter = 0
        self.log_limit = 500
        # 在脚本运行的目录下创建(或覆盖)一个日志文件
        self.log_file = open("bridge_log.txt", "w")
        # --- 结束新增代码 ---
        rospy.loginfo("YOPO to MAVROS bridge is running.")

    def yopo_cmd_callback(self, msg):
        """
        当收到来自 YOPO 的 PositionCommand 消息时，此函数被调用。
        它会将消息转换为 PoseStamped 并发布。
        """
        rospy.loginfo("=========================================================")
        rospy.loginfo("RECEIVED Original PositionCommand from /yopo/pos_cmd:")
        rospy.loginfo(str(msg)) # str(msg) 会将整个消息的内容格式化为字符串并打印
        #开始转化
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"

        # 直接复制位置信息
        pose_msg.pose.position.x = msg.position.x
        pose_msg.pose.position.y = msg.position.y
        pose_msg.pose.position.z = 2.0  # 固定高度为 2 米

        # 将 YOPO 的偏航角 (yaw) 转换为四元数
        q = quaternion_from_euler(0, 0, msg.yaw)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        rospy.loginfo("----------------- CONVERTED TO -----------------")
        rospy.loginfo("SENDING Converted PoseStamped to /mavros/setpoint_position/local:")
        rospy.loginfo(str(pose_msg)) # 同样，打印转换后的完整消息内容
        rospy.loginfo("=========================================================\n") # 加一个换行方便阅读
# --- 新增代码：写入日志文件 ---
        if self.log_counter < self.log_limit:
            log_content = (
                f"--- Message #{self.log_counter + 1} ---\n"
                f"RECEIVED Original PositionCommand from /yopo/pos_cmd:\n{str(msg)}\n"
                f"----------------- CONVERTED TO -----------------\n"
                f"SENDING Converted PoseStamped to /mavros/setpoint_position/local:\n{str(pose_msg)}\n"
                f"=========================================================\n\n"
            )
            self.log_file.write(log_content)
            self.log_counter += 1
            if self.log_counter == self.log_limit:
                rospy.loginfo("Reached 500 messages, closing bridge_log.txt.")
                self.log_file.close()
        # --- 结束新增代码 ---
        # 发布转换后的消息给 MAVROS
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        # 确保 control_msg.py 能被找到
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        bridge = YopoBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass