import numpy as np
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from typing import Tuple, List
import time
import pinocchio
from loguru import logger

from .. import fast

XyzXyzw = fast.XyzXyzw

def xyz_xyzw_to_SE3(xyz_quat: XyzXyzw):
    xyz, quat = xyz_quat
    return pinocchio.SE3(
        R.from_quat(quat).as_matrix(),
        xyz
    )

class PinoIkSolver:
    def __init__(
        self,
        urdf: str,
    ):
        if urdf == '[sample]':
            self.model = pinocchio.buildSampleModelManipulator()
        else:
            self.model = pinocchio.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        # cache
        self.cache_frames = [(x.name, x.type) for x in self.model.frames]
        self.cache_joints = [x for x in self.model.names]

    def joint_name_to_qid(self, joint: str):
        return self.model.joints[self.model.getJointId(joint)].idx_q

    def get_joint_chain_idx_jq(self, base_j: str, end_j: str):
        base_j_id = self.model.getJointId(base_j)
        end_j_id = self.model.getJointId(end_j)
        cur = end_j_id
        chain_jid = [cur]
        while cur != base_j_id:
            cur = self.model.parents[cur]
            chain_jid.append(cur)
        chain_jid.reverse()
        # chain_q_indices = [self.model.joints[i].idx_q for i in chain]
        chain_qid = [self.model.joints[i].idx_q for i in chain_jid]
        return chain_jid, chain_qid

    # return quat[xyzw]
    def get_tcp(self, q: np.array, base_link: tuple, tcp_name: tuple):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        # uni<-base
        base_pose = self.data.oMf[self.model.getFrameId(*base_link)]
        tcp_pose = self.data.oMf[self.model.getFrameId(*tcp_name)]
        base4tcp = base_pose.actInv(tcp_pose)
        return base4tcp.translation, R.from_matrix(base4tcp.rotation).as_quat()

    def solve_ik(
        self,
        move_joints: List[str],
        loss: List[tuple],
        ref_q: np.array=None,
        verbose=False,
        minimize_options=None,
    ):
        """
        move: THJ5 -> THJ1 (5 in total)
        自定义损失函数 tuple[one of these]:
        - "frame_target" target base <- target end
            - target base: name, type
            - target end: name, type
            - target transform: (xyz, xyzw)
            - 位置模长系数
            - 姿态模长系数
        - "frame_del" not yet implemented
            - target base: name, type
            - target end: name, type
            - 位置模长系数
            - 姿态模长系数
        - "joint_del"
        - joint
            - 系数
        ex:
        loss=[
            ("frame_target", ('ra_base_link', pinocchio.FrameType.BODY),
             ('rh_palm', pinocchio.FrameType.BODY), (xyz, xyzw), 1.0, 1.0),
            ("frame_del", ('ra_shoulder_pan_joint', pinocchio.FrameType.JOINT),
             ('ra_shoulder_lift_joint', pinocchio.FrameType.JOINT), 0.1, 1.0),
        ],

        """
        if ref_q is None:
            ref_q = pinocchio.neutral(self.model)
        move_joints_qid = [self.joint_name_to_qid(j) for j in move_joints]

        # 缓存
        def get_two_frame_trans_pino_se3(base_f, end_f):
            base_f_id = self.model.getFrameId(*base_f)
            end_f_id = self.model.getFrameId(*end_f)
            base_pose = self.data.oMf[base_f_id]
            end_pose = self.data.oMf[end_f_id]
            base4end = base_pose.inverse() * end_pose
            return base4end

        pinocchio.forwardKinematics(self.model, self.data, ref_q)
        pinocchio.updateFramePlacements(self.model, self.data)
        two_frame_origin_trans_cache = {
            (nametype1, nametype2): fast.xyz_xyzw_from_matrix(np.array(get_two_frame_trans_pino_se3(nametype1, nametype2)))
            for nametype1, nametype2 in [(term[1], term[2]) for term in loss if term[0] == "frame_del"]
        }

        # [todo] 测量延迟
        def obj(q_sub, verbose=False):
            q = ref_q.copy() # 将不可动关节考虑进来
            q[move_joints_qid] = q_sub
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)
            def frame_target(tar_base_f, tar_end_f, target: XyzXyzw, pos_coef, rot_coef):
                tar_base_f4tar_end_f_des = xyz_xyzw_to_SE3(target)
                tar_base_f4tar_end_f_real = get_two_frame_trans_pino_se3(tar_base_f, tar_end_f)
                tar_end_f_des4tar_end_f_real = tar_base_f4tar_end_f_des.inverse() * tar_base_f4tar_end_f_real
                vec = pinocchio.log(tar_end_f_des4tar_end_f_real).vector
                return pos_coef * norm(vec[:3]) ** 2 + rot_coef * norm(vec[3:]) ** 2

            def frame_del(tar_base_f, tar_end_f, pos_coef, rot_coef):
                return frame_target(tar_base_f, tar_end_f, two_frame_origin_trans_cache[(tar_base_f, tar_end_f)], pos_coef, rot_coef)

            def joint_del(joint_name, coef):
                qid = self.joint_name_to_qid(joint_name)
                q_del = q[qid] - ref_q[qid]
                return coef * abs(q_del) ** 2

            fn_map = { "frame_target": frame_target, "frame_del": frame_del, "joint_del": joint_del }

            if verbose:
                for i, term in enumerate(loss):
                    thisloss = fn_map[term[0]](*term[1:])
                    logger.info(f"term {i}: {thisloss}")
            tot = sum([fn_map[term[0]](*term[1:]) for term in loss])
            return tot


        bounds = [(self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i]) for i in move_joints_qid]
        # minimize: 优化方法为
        iter_cnt = [0]
        def iter_callback(xk):
            iter_cnt[0] += 1

        result = minimize(obj, ref_q[move_joints_qid], bounds=bounds, callback=iter_callback, options=minimize_options)
        q = ref_q.copy()
        q[move_joints_qid] = result.x
        if verbose:
            obj(result.x, verbose=True)
            print(f"iter_cnt: {iter_cnt[0]}")
        return q

if __name__ == "__main__":
    solver = PinoIkSolver('[sample]')
    print(solver.get_tcp(np.zeros(6), ('shoulder1_joint', pinocchio.FrameType.JOINT), ('wrist2_joint', pinocchio.FrameType.JOINT)))

    q = solver.solve_ik(
        move_joints=['shoulder1_joint', 'shoulder2_joint',  'shoulder3_joint', 'elbow_joint', 'wrist1_joint', 'wrist2_joint'],
        loss=[
            ("frame_target",
                ('shoulder1_joint', pinocchio.FrameType.JOINT),
                ('wrist2_joint', pinocchio.FrameType.JOINT),
                (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])),
                1.0, 1.0
            ),
        ],
        minimize_options={
            "maxiter": 10,
        },
        verbose=True,
    )
    print(q)
    print(solver.get_tcp(q, ('shoulder1_joint', pinocchio.FrameType.JOINT), ('wrist2_joint', pinocchio.FrameType.JOINT)))
