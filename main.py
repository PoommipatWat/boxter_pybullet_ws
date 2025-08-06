import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

import numpy as np

import time

#     -> torso_t0    -(f)-> right_torso_arm_mount 
# -(r)-> right_s0 -(r)-> right_s1    -(r)-> right_e0
# -(r)-> right_e1 -(r)-> right_w0    -(r)-> right_w1
# -(r)-> right_w2 -(f)-> right_hand  -(f)-> right_gripper_base
# -(f)-> right_endpoint

# youtube : https://www.youtube.com/watch?v=iidcy8RhOmY

def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def rpy_to_rotation_matrix_fixed_axis(rpy):
    R_z = rotation_matrix_z(rpy[2])
    R_y = rotation_matrix_y(rpy[1])
    R_x = rotation_matrix_x(rpy[0])
    return R_z @ R_y @ R_x

def rot_to_quaternion(R):
   e4 = (1/2.0) * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
   
   if np.abs(e4) < 1e-6:
       e1_candidate = (1/2.0) * np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1)
       e2_candidate = (1/2.0) * np.sqrt(-R[0, 0] + R[1, 1] - R[2, 2] + 1)
       e3_candidate = (1/2.0) * np.sqrt(-R[0, 0] - R[1, 1] + R[2, 2] + 1)
       
       max_val = max(e1_candidate, e2_candidate, e3_candidate)
       
       if max_val == e1_candidate:
           e1 = e1_candidate
           e2 = (R[1, 0] + R[0, 1]) / (4 * e1)
           e3 = (R[2, 0] + R[0, 2]) / (4 * e1)
           e4 = (R[2, 1] - R[1, 2]) / (4 * e1)
       elif max_val == e2_candidate:
           e2 = e2_candidate
           e1 = (R[1, 0] + R[0, 1]) / (4 * e2)
           e3 = (R[2, 1] + R[1, 2]) / (4 * e2)
           e4 = (R[0, 2] - R[2, 0]) / (4 * e2)
       else:
           e3 = e3_candidate
           e1 = (R[2, 0] + R[0, 2]) / (4 * e3)
           e2 = (R[2, 1] + R[1, 2]) / (4 * e3)
           e4 = (R[1, 0] - R[0, 1]) / (4 * e3)
   else:
       e1 = (R[2, 1] - R[1, 2]) / (4 * e4)
       e2 = (R[0, 2] - R[2, 0]) / (4 * e4)
       e3 = (R[1, 0] - R[0, 1]) / (4 * e4)
   return np.array([e1, e2, e3, e4])

def create_homogeneous_transformation(joints, move_angles=None):
    T_array = []

    T_total = np.eye(4)

    for i, joint in enumerate(joints):
        T = np.eye(4)
        T[:3, 3] = joint['position']
        T[:3, :3] = rpy_to_rotation_matrix_fixed_axis(joint['orientation'])

        if move_angles != None:
            T[:3, :3] = T[:3, :3] @ rotation_matrix_z(list(move_angles.values())[i])

        T_total = T_total @ T

        T_array.append(T_total)

    return T_array

def load_joint_correct(robot_id, joint_name: str):
    # หาจุดสิ้นสุดของแขน (end effector) จาก PyBullet
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        child_link_name = joint_info[12].decode()
        
        # หา end effector link
        if child_link_name == "right_gripper":
            link_state = p.getLinkState(robot_id, i)
            
            data_info = {}
            data_info['id'] = i
            data_info['name'] = child_link_name
            data_info['position'] = np.round(link_state[4], 5)      # World coordinate position
            data_info['orientation'] = np.round(link_state[5], 5)   # World coordinate orientation
    
            return data_info

def get_joint_states(robot_id):
    """ดึงข้อมูลสถานะของ revolute joints เท่านั้น"""
    num_joints = p.getNumJoints(robot_id)
    joint_positions = []
    joint_velocities = []
    revolute_joint_indices = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        joint_name = joint_info[1].decode()
        
        # เฉพาะ revolute joints เท่านั้น
        if joint_type == p.JOINT_REVOLUTE:
            joint_state = p.getJointState(robot_id, i)
            joint_positions.append(joint_state[0])  # ตำแหน่ง
            joint_velocities.append(joint_state[1]) # ความเร็ว
            revolute_joint_indices.append(i)

    return joint_positions, joint_velocities, revolute_joint_indices

def find_end_effector_link_index(robot_id, target_link_name="right_gripper"):
    """หา index ของ end effector link"""
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        child_link_name = joint_info[12].decode()
        
        if child_link_name == target_link_name:
            return i
    
    return -1  # ถ้าหาไม่เจอ

def find_J(T):
    J = []

    Xp = T[-1][:3, 3]

    for i in T:
        r = i[:3, 2]
        o = i[:3, 3]

        linear_part = np.cross(r, (Xp - o))
        angular_part = r

        Ji = np.concatenate([linear_part, angular_part])

        J.append(Ji)
    
    return np.column_stack(J)


def main():

    # เชื่อมต่อ PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # โหลดหุ่นยนต์ Baxter
    robot_id = load_robot_description("baxter_description", useFixedBase=True)

    urdf_joint_base = [
        {"name": "torso_t0",                 "position": [0, 0, 0],                          "orientation": [0, 0, 0]},
        {"name": "right_torso_arm_mount",    "position": [0.024645, -0.219645, 0.118588],    "orientation": [0, 0, -0.7854]},
    ]
    
    urdf_joint_arm = [
        {"name": "right_s0",                 "position": [0.055695, 0, 0.011038],            "orientation": [0, 0, 0]},
        {"name": "right_s1",                 "position": [0.069, 0, 0.27035],                "orientation": [-1.57079632679, 0, 0]},
        {"name": "right_e0",                 "position": [0.102, 0, 0],                      "orientation": [1.57079632679, 0, 1.57079632679]},
        {"name": "right_e1",                 "position": [0.069, 0, 0.26242],                "orientation": [-1.57079632679, -1.57079632679, 0]},
        {"name": "right_w0",                 "position": [0.10359, 0, 0],                    "orientation": [1.57079632679, 0, 1.57079632679]},
        {"name": "right_w1",                 "position": [0.01, 0, 0.2707],                  "orientation": [-1.57079632679, -1.57079632679, 0]},
        {"name": "right_w2",                 "position": [0.115975, 0, 0],                   "orientation": [1.57079632679, 0, 1.57079632679]},
    ]

    urdf_joints_endefector = [
        {"name": "right_hand",               "position": [0, 0, 0.11355],                    "orientation": [0, 0, 0]},
        {"name": "right_gripper_base",       "position": [0, 0, 0],                          "orientation": [0, 0, 0]},
        {"name": "right_endpoint",           "position": [0, 0, 0.025],                      "orientation": [0, 0, 0]}
    ]

    urdf_all = urdf_joint_base + urdf_joint_arm + urdf_joints_endefector

    while True:
        target_positions = { #สุ่มโดยใช้ช่วง limit แกนหมุนของ URDF 
            'torso_t0': 0,
            'right_torso_arm_mount':0,
            'right_s0': np.random.uniform(-1.70167993878, 1.70167993878),
            'right_s1': np.random.uniform(-2.147, 1.047),
            'right_e0': np.random.uniform(-3.05417993878, 3.05417993878),
            'right_e1': np.random.uniform(-0.05, 2.618),
            'right_w0': np.random.uniform(-3.059, 3.059),
            'right_w1': np.random.uniform(-1.57079632679, 2.094),
            'right_w2': np.random.uniform(-3.059, 3.059),
            'right_hand':0,
            'right_gripper_base':0,
            'right_endpoint':0
        }

        T_base_end = create_homogeneous_transformation(urdf_all, target_positions)

        print(f"Position Calculate : {np.round(T_base_end[-1][:3, 3], 5)}")
        print(f"Orientation Calculate : {np.round(rot_to_quaternion(T_base_end[-1][:3, :3]), 5)}")

        # สร้างบอล ไม่เอา collision เพราะแขนจะชนแล้วไปไม่ถึง
        ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        ball_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ball_visual,
            basePosition=T_base_end[-1][:3, 3]
        )

        # Move the joints
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            joint_name = info[1].decode()
            joint_type = info[2]

            if joint_name in target_positions and joint_type == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_positions[joint_name],
                    force=100000.0
                )

        # Run simulation
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        joint_data = load_joint_correct(robot_id, "right_w2")
        print(f"Position Pybullet : {joint_data['position']}")
        print(f"Orientation Pybullet : {joint_data['orientation']}")

        p.removeBody(ball_id)

        #----------------------------- Jacobian ------------------------------#
        
        # ดึงข้อมูลสถานะ revolute joints เท่านั้น
        joint_positions, joint_velocities, revolute_joint_indices = get_joint_states(robot_id)
        
        # กำหนดความเร่ง (ถ้าไม่มีให้ใส่ 0)
        joint_accelerations = [0.0] * len(joint_positions)
        
        # หา index ของ end effector
        end_effector_index = find_end_effector_link_index(robot_id, "right_wrist")
        
        if end_effector_index == -1:
            print("ไม่พบ end effector link!")
            continue

        jacobian_linear, jacobian_angular = p.calculateJacobian(
            bodyUniqueId=robot_id,                    # ID ของหุ่นยนต์
            linkIndex=end_effector_index,             # Index ของ link ปลาย (end-effector) 
            localPosition=[0, 0, 0],                  # ตำแหน่งใน local frame ของ link
            objPositions=joint_positions,             # ตำแหน่ง revolute joints เท่านั้น
            objVelocities=joint_velocities,           # ความเร็ว revolute joints เท่านั้น
            objAccelerations=joint_accelerations      # ความเร่ง revolute joints เท่านั้น
        )
        

        jac_linear = np.array(jacobian_linear)
        jac_angular = np.array(jacobian_angular)

        full_jacobian = np.vstack([jac_linear, jac_angular])

        non_zero_cols = []
        for i in range(full_jacobian.shape[1]):
            if np.any(np.abs(full_jacobian[:, i]) > 1e-6):
                non_zero_cols.append(i)

        if len(non_zero_cols) > 0:
            active_jacobian = full_jacobian[:, non_zero_cols]
            print("Active Jacobian:")
            print(np.round(active_jacobian, 4))

        T_base_use = T_base_end[2:-3]
        cal_J = find_J(T_base_use)

        print("Calculate Jacobian:")
        print(np.round(cal_J,4))
        
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()