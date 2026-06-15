import numpy as np
from scipy.linalg import eigh, cholesky, solve
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Callable
from .timber_constitutive import TimberOrthotropicConstitutive, TimberBeamElement


class PagodaFEAModel:
    """
    应县木塔有限元模型
    简化的多层框架模型，考虑木材各向异性
    """

    def __init__(self, timber_properties: Dict[str, float]):
        """
        初始化有限元模型

        Args:
            timber_properties: 木材材料参数
        """
        self.constitutive = TimberOrthotropicConstitutive(timber_properties)
        self.nodes: List[np.ndarray] = []
        self.elements: List[TimberBeamElement] = []
        self.node_dof_map: Dict[int, np.ndarray] = {}
        self.n_dofs: int = 0
        self.boundary_conditions: Dict[int, np.ndarray] = {}
        self.K: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.natural_frequencies: Optional[np.ndarray] = None
        self.mode_shapes: Optional[np.ndarray] = None
        self.floor_heights: np.ndarray = np.array([9.23, 17.73, 25.53, 32.73, 39.23])
        self.floor_diameters: np.ndarray = np.array([30.27, 25.80, 22.50, 19.80, 17.50])
        self.n_floors = 5
        self.columns_per_floor = 24
        self.beams_per_floor = 48

    def build_model(self):
        """构建木塔有限元模型"""
        self._create_nodes()
        self._create_column_elements()
        self._create_beam_elements()
        self._assemble_global_matrices()
        self._apply_boundary_conditions()

    def _create_nodes(self):
        """创建节点 - 每层创建24个柱节点"""
        node_id = 0

        for floor_idx in range(self.n_floors + 1):
            z = 0 if floor_idx == 0 else self.floor_heights[floor_idx - 1]
            radius = 15.0 if floor_idx == 0 else self.floor_diameters[floor_idx - 1] / 2

            for i in range(self.columns_per_floor):
                angle = 2 * np.pi * i / self.columns_per_floor
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                self.nodes.append(np.array([x, y, z]))
                self.node_dof_map[node_id] = np.arange(node_id * 6, (node_id + 1) * 6)
                node_id += 1

        self.n_nodes = node_id
        self.n_dofs = self.n_nodes * 6

    def _create_column_elements(self):
        """创建柱单元"""
        column_section = {
            'width': 0.6,
            'height': 0.6,
            'A': 0.6 * 0.6,
            'Ixx': 0.6 * 0.6 ** 3 / 12,
            'Iyy': 0.6 * 0.6 ** 3 / 12,
            'Izz': 0.6 * 0.6 ** 3 / 12
        }

        for floor_idx in range(self.n_floors):
            for col_idx in range(self.columns_per_floor):
                node_i = floor_idx * self.columns_per_floor + col_idx
                node_j = (floor_idx + 1) * self.columns_per_floor + col_idx

                element = TimberBeamElement(
                    self.nodes[node_i],
                    self.nodes[node_j],
                    self.constitutive,
                    column_section
                )
                self.elements.append(element)

    def _create_beam_elements(self):
        """创建梁单元 - 每层创建内外两圈梁"""
        beam_section = {
            'width': 0.4,
            'height': 0.8,
            'A': 0.4 * 0.8,
            'Ixx': 0.4 * 0.8 ** 3 / 12,
            'Iyy': 0.8 * 0.4 ** 3 / 12,
            'Izz': 0.4 * 0.8 ** 3 / 12
        }

        for floor_idx in range(1, self.n_floors + 1):
            base_node = floor_idx * self.columns_per_floor

            for i in range(self.columns_per_floor):
                node_i = base_node + i
                node_j = base_node + (i + 1) % self.columns_per_floor

                element = TimberBeamElement(
                    self.nodes[node_i],
                    self.nodes[node_j],
                    self.constitutive,
                    beam_section
                )
                self.elements.append(element)

            for i in range(0, self.columns_per_floor, 2):
                node_i = base_node + i
                node_j = base_node + (i + self.columns_per_floor // 2) % self.columns_per_floor

                element = TimberBeamElement(
                    self.nodes[node_i],
                    self.nodes[node_j],
                    self.constitutive,
                    beam_section
                )
                self.elements.append(element)

    def _assemble_global_matrices(self):
        """组装整体刚度矩阵和质量矩阵"""
        K = lil_matrix((self.n_dofs, self.n_dofs))
        M = lil_matrix((self.n_dofs, self.n_dofs))

        for element in self.elements:
            k_e = element.get_global_stiffness()
            m_e = element.get_global_mass()

            nodes = [self.nodes.index(element.node_i), self.nodes.index(element.node_j)]
            dofs = []
            for node in nodes:
                dofs.extend(self.node_dof_map[node].tolist())

            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    K[di, dj] += k_e[i, j]
                    M[di, dj] += m_e[i, j]

        self.K = K.tocsr()
        self.M = M.tocsr()

    def _apply_boundary_conditions(self):
        """施加边界条件 - 底部固定"""
        for node_id in range(self.columns_per_floor):
            dofs = self.node_dof_map[node_id]
            self.boundary_conditions[node_id] = dofs

        fixed_dofs = []
        for dofs in self.boundary_conditions.values():
            fixed_dofs.extend(dofs.tolist())

        self.fixed_dofs = np.array(fixed_dofs)
        self.free_dofs = np.setdiff1d(np.arange(self.n_dofs), self.fixed_dofs)

    def compute_modal_analysis(self, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        模态分析

        Args:
            n_modes: 计算的模态阶数

        Returns:
            natural_frequencies: 固有频率 (Hz)
            mode_shapes: 振型矩阵
        """
        K_ff = self.K[self.free_dofs, :][:, self.free_dofs]
        M_ff = self.M[self.free_dofs, :][:, self.free_dofs]

        K_dense = K_ff.toarray()
        M_dense = M_ff.toarray()

        eigenvalues, eigenvectors = eigh(K_dense, M_dense, subset_by_index=[0, n_modes - 1])

        omega = np.sqrt(np.maximum(eigenvalues, 0))
        frequencies = omega / (2 * np.pi)

        mode_shapes_full = np.zeros((self.n_dofs, n_modes))
        mode_shapes_full[self.free_dofs, :] = eigenvectors

        for i in range(n_modes):
            max_val = np.max(np.abs(mode_shapes_full[:, i]))
            if max_val > 0:
                mode_shapes_full[:, i] /= max_val

        self.natural_frequencies = frequencies
        self.mode_shapes = mode_shapes_full

        return frequencies, mode_shapes_full

    def build_damping_matrix(self, damping_ratio: float = 0.02) -> np.ndarray:
        """
        构建瑞利阻尼矩阵

        C = alpha * M + beta * K

        Args:
            damping_ratio: 阻尼比

        Returns:
            C: 阻尼矩阵
        """
        if self.natural_frequencies is None:
            self.compute_modal_analysis()

        omega1 = 2 * np.pi * self.natural_frequencies[0]
        omega2 = 2 * np.pi * self.natural_frequencies[min(2, len(self.natural_frequencies) - 1)]

        alpha = 2 * damping_ratio * omega1 * omega2 / (omega1 + omega2)
        beta = 2 * damping_ratio / (omega1 + omega2)

        self.C = alpha * self.M + beta * self.K

        return self.C

    def _newmark_beta_solve(self, F: np.ndarray, dt: float,
                             u0: Optional[np.ndarray] = None,
                             v0: Optional[np.ndarray] = None,
                             gamma: float = 0.5,
                             beta: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Newmark-beta法求解动力时程

        Args:
            F: 荷载矩阵 [n_dofs, n_timesteps]
            dt: 时间步长
            u0: 初始位移
            v0: 初始速度
            gamma, beta: Newmark参数

        Returns:
            u: 位移时程
            v: 速度时程
            a: 加速度时程
        """
        n_steps = F.shape[1]

        if u0 is None:
            u0 = np.zeros(self.n_dofs)
        if v0 is None:
            v0 = np.zeros(self.n_dofs)

        u = np.zeros((self.n_dofs, n_steps))
        v = np.zeros((self.n_dofs, n_steps))
        a = np.zeros((self.n_dofs, n_steps))

        u[:, 0] = u0
        v[:, 0] = v0

        K_ff = self.K[self.free_dofs, :][:, self.free_dofs]
        M_ff = self.M[self.free_dofs, :][:, self.free_dofs]
        C_ff = self.C[self.free_dofs, :][:, self.free_dofs]
        F_ff = F[self.free_dofs, :]

        a0_ff = spsolve(K_ff, F_ff[:, 0])

        u_ff = np.zeros((len(self.free_dofs), n_steps))
        v_ff = np.zeros((len(self.free_dofs), n_steps))
        a_ff = np.zeros((len(self.free_dofs), n_steps))

        u_ff[:, 0] = u0[self.free_dofs]
        v_ff[:, 0] = v0[self.free_dofs]
        a_ff[:, 0] = a0_ff

        K_hat = K_ff + gamma / (beta * dt) * C_ff + 1 / (beta * dt ** 2) * M_ff

        for step in range(n_steps - 1):
            u_pred = u_ff[:, step] + dt * v_ff[:, step] + (dt ** 2 / 2) * (1 - 2 * beta) * a_ff[:, step]
            v_pred = v_ff[:, step] + dt * (1 - gamma) * a_ff[:, step]

            F_hat = F_ff[:, step + 1] - C_ff @ v_pred - K_ff @ u_pred

            delta_u = spsolve(K_hat, F_hat)

            delta_v = gamma / (beta * dt) * delta_u - gamma / beta * v_ff[:, step] + dt * (1 - gamma / (2 * beta)) * a_ff[:, step]
            delta_a = 1 / (beta * dt ** 2) * delta_u - 1 / (beta * dt) * v_ff[:, step] - 1 / (2 * beta) * a_ff[:, step]

            u_ff[:, step + 1] = u_pred + delta_u
            v_ff[:, step + 1] = v_pred + delta_v
            a_ff[:, step + 1] = a_ff[:, step] + delta_a

        u[self.free_dofs, :] = u_ff
        v[self.free_dofs, :] = v_ff
        a[self.free_dofs, :] = a_ff

        return u, v, a

    def _modal_superposition(self, F: np.ndarray, dt: float,
                              n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        模态叠加法求解动力时程

        Args:
            F: 荷载矩阵 [n_dofs, n_timesteps]
            dt: 时间步长
            n_modes: 使用的模态阶数

        Returns:
            u: 位移时程
            v: 速度时程
            a: 加速度时程
        """
        if self.natural_frequencies is None:
            self.compute_modal_analysis(n_modes)

        n_steps = F.shape[1]
        omega = 2 * np.pi * self.natural_frequencies[:n_modes]
        modes = self.mode_shapes[:, :n_modes]

        M_modal = np.diag(modes.T @ self.M @ modes)
        K_modal = np.diag(modes.T @ self.K @ modes)
        C_modal = np.diag(modes.T @ self.C @ modes)

        F_modal = modes.T @ F

        xi_modal = C_modal / (2 * omega * M_modal)

        u_modal = np.zeros((n_modes, n_steps))
        v_modal = np.zeros((n_modes, n_steps))
        a_modal = np.zeros((n_modes, n_steps))

        for i in range(n_modes):
            omega_i = omega[i]
            xi_i = xi_modal[i]
            F_i = F_modal[i, :] / M_modal[i]

            omega_d = omega_i * np.sqrt(1 - xi_i ** 2)

            for step in range(1, n_steps):
                t = step * dt
                f_prev = F_i[step - 1]
                f_curr = F_i[step]

                u_prev = u_modal[i, step - 1]
                v_prev = v_modal[i, step - 1]

                A = (f_curr - f_prev) / dt
                B = f_prev - A * (t - dt)

                u_p = (A * t + B - 2 * xi_i * A / omega_i) / omega_i ** 2
                u_p_prev = (A * (t - dt) + B - 2 * xi_i * A / omega_i) / omega_i ** 2

                C = u_prev - u_p_prev
                D = (v_prev + xi_i * omega_i * C - A / omega_i ** 2) / omega_d

                e_term = np.exp(-xi_i * omega_i * dt)
                cos_term = np.cos(omega_d * dt)
                sin_term = np.sin(omega_d * dt)

                u_modal[i, step] = e_term * (C * cos_term + D * sin_term) + u_p
                v_modal[i, step] = -xi_i * omega_i * u_modal[i, step] + \
                                   e_term * omega_d * (-C * sin_term + D * cos_term) + A / omega_i ** 2

            a_modal[i, :] = F_i - 2 * xi_i * omega_i * v_modal[i, :] - omega_i ** 2 * u_modal[i, :]

        u = modes @ u_modal
        v = modes @ v_modal
        a = modes @ a_modal

        return u, v, a

    def solve_dynamic_response(self, loads: Dict[int, np.ndarray],
                                t: np.ndarray,
                                damping_ratio: float = 0.02,
                                method: str = 'newmark') -> Dict:
        """
        求解结构动力响应

        Args:
            loads: 各节点荷载 {node_id: force_time_history}
            t: 时间向量
            damping_ratio: 阻尼比
            method: 'newmark' 或 'modal'

        Returns:
            results: 计算结果字典
        """
        self.build_damping_matrix(damping_ratio)

        dt = t[1] - t[0]
        n_steps = len(t)

        F = np.zeros((self.n_dofs, n_steps))

        for node_id, force in loads.items():
            dofs = self.node_dof_map[node_id]
            if len(force) == n_steps:
                F[dofs[0], :] = force
            elif len(force.shape) == 2 and force.shape[0] == 3:
                for i in range(3):
                    F[dofs[i], :] = force[i, :]

        if method == 'newmark':
            u, v, a = self._newmark_beta_solve(F, dt)
        else:
            u, v, a = self._modal_superposition(F, dt)

        floor_displacements = self._extract_floor_displacements(u)
        floor_accelerations = self._extract_floor_accelerations(a)
        element_stresses = self._compute_element_stresses(u)

        results = {
            'time': t,
            'displacement': u,
            'velocity': v,
            'acceleration': a,
            'floor_displacements': floor_displacements,
            'floor_accelerations': floor_accelerations,
            'element_stresses': element_stresses,
            'natural_frequencies': self.natural_frequencies,
            'mode_shapes': self.mode_shapes.tolist() if self.mode_shapes is not None else None
        }

        return results

    def _extract_floor_displacements(self, u: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """提取各层位移"""
        floor_disp = {}

        for floor_idx in range(self.n_floors):
            base_node = (floor_idx + 1) * self.columns_per_floor

            disp_x = []
            disp_y = []
            disp_z = []

            for col_idx in range(self.columns_per_floor):
                node_id = base_node + col_idx
                dofs = self.node_dof_map[node_id]
                disp_x.append(u[dofs[0], :])
                disp_y.append(u[dofs[1], :])
                disp_z.append(u[dofs[2], :])

            floor_disp[floor_idx + 1] = {
                'x': np.mean(np.array(disp_x), axis=0),
                'y': np.mean(np.array(disp_y), axis=0),
                'z': np.mean(np.array(disp_z), axis=0),
                'max_x': np.max(np.array(disp_x), axis=0),
                'max_y': np.max(np.array(disp_y), axis=0),
                'max_z': np.max(np.array(disp_z), axis=0)
            }

        return floor_disp

    def _extract_floor_accelerations(self, a: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """提取各层加速度"""
        floor_acc = {}

        for floor_idx in range(self.n_floors):
            base_node = (floor_idx + 1) * self.columns_per_floor

            acc_x = []
            acc_y = []

            for col_idx in range(self.columns_per_floor):
                node_id = base_node + col_idx
                dofs = self.node_dof_map[node_id]
                acc_x.append(a[dofs[0], :])
                acc_y.append(a[dofs[1], :])

            floor_acc[floor_idx + 1] = {
                'x': np.mean(np.array(acc_x), axis=0),
                'y': np.mean(np.array(acc_y), axis=0),
                'max_x': np.max(np.array(acc_x), axis=0),
                'max_y': np.max(np.array(acc_y), axis=0)
            }

        return floor_acc

    def _compute_element_stresses(self, u: np.ndarray) -> List[Dict]:
        """计算单元应力"""
        element_stresses = []

        for elem_idx, element in enumerate(self.elements):
            node_i = self.nodes.index(element.node_i)
            node_j = self.nodes.index(element.node_j)

            dofs = []
            for node in [node_i, node_j]:
                dofs.extend(self.node_dof_map[node].tolist())

            u_e = u[dofs, :]

            u_local = element.transformation_matrix @ u_e

            n_steps = u.shape[1]
            max_stress = 0

            for step in range(n_steps):
                du_local = u_local[:, step]

                strain = np.zeros(6)
                strain[0] = (du_local[6] - du_local[0]) / element.length

                stress = self.constitutive.compute_stress(strain)
                max_stress = max(max_stress, np.max(np.abs(stress)))

            element_stresses.append({
                'element_id': elem_idx,
                'max_stress': max_stress / 1e6,
                'node_i': node_i,
                'node_j': node_j,
                'floor': int(node_i / self.columns_per_floor)
            })

        return element_stresses
