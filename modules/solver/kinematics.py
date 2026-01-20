import numpy as np
from .exponential import Exponential
from ..displayer import OrigamiVisualizer
from scipy.interpolate import interp1d
from .util import compute_constraint_matrix

class Kinematics:
    def __init__(self, kirigami_obj, k_raw=None, theta_init_raw = None, theta_equil_raw = None, face_colors = None, reference_frame =None):
        """
        Initializes the state for origami dynamics systaem.
        """
        self.kirigami = kirigami_obj
        self.theta_init= self.kirigami.get_internal_theta(self.kirigami.get_final_desi(theta_init_raw))
        self.theta_equil =  self.kirigami.get_internal_theta(self.kirigami.get_final_desi(theta_equil_raw))
        self.face_colors = face_colors
        self.k = self.kirigami.stiff(k_raw)
        self.H = np.diag(self.k)

        # Geometry and screw structure
        self.closed_loop_home_screw = self.kirigami.screw_loops
        self.closed_loop_crease_idx = self.kirigami.index_screws
        self.num_creases = len(self.theta_equil)
        self.num_loops = len(self.closed_loop_home_screw)

        self.step_size_threshold = 1e-9
        self.residual_threshold = 1e-9
        self.max_diff_threshold = 1e-5
        self.MAX_ITERATION = 5000
        self.init_step_size = np.pi/180

        self.num_interp_frames = 200
        self.fps = 30
        self.play_speed = 1.0
        
        self.reference_frame = reference_frame
        self.angle_history = None


    def _residual_at_step(self, theta_list):
        violations = []

        for screw, crease_idx in zip(self.closed_loop_home_screw, self.closed_loop_crease_idx):
            # Extract joint angles for this loop
            theta_loop = theta_list[crease_idx]

            # Compute loop closure transform
            if screw.shape[0] == 3:
                R_loop = Exponential.frot_space(np.eye(3), screw, theta_loop)
                a, b, c = R_loop[2, 1], R_loop[0, 2], R_loop[1, 0]
                violations.append(np.array([a, b, c]))
            elif screw.shape[0] == 6:
                T_loop = Exponential.fkin_space(np.eye(4), screw, theta_loop)
                R, p = T_loop[:3, :3], T_loop[:3, 3]
                a, b, c = R[2, 1], R[0, 2], R[1, 0]
                violations.append(np.hstack([a, b, c, p]))
            else:
                raise ValueError("Screw must have 3 or 6 rows.")

        return np.concatenate(violations)[:, np.newaxis]
    
    def _find_equilibrium(self):
        theta = self.theta_init
        c = self.init_step_size
        angle_history = []
        i = 0
        prev_ref_angle = theta[0]
        ref_angle = prev_ref_angle
        while c > self.step_size_threshold and i < self.MAX_ITERATION:
            i+=1
            C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)
            r = self._residual_at_step(theta)
            d = self.k * (theta - self.theta_equil)
            d = d[:, np.newaxis]

            C_rank = np.linalg.matrix_rank(C)
            if C_rank == C.shape[0]:
                H_1 = np.linalg.inv(self.H)
                G = H_1 @ C.T @ np.linalg.inv(C @ H_1 @ C.T)
                delta_theta = - (H_1 - G @ C @ H_1) @ d - G @ r
            else:
                dr = np.vstack((d, r))
                hcc0 = np.hstack((self.H, C.T))
                hcc0 = np.vstack((hcc0, np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))))
                delta_theta = - (np.linalg.pinv(hcc0) @ dr)[:len(theta), :]
            if i>2 and delta_theta[0]*(ref_angle - prev_ref_angle) < 0:
                c = c/2
            prev_ref_angle = ref_angle
            
            if np.max(np.abs(delta_theta)) > 0.0:
                theta = theta + delta_theta.flatten() * c / np.max(np.abs(delta_theta))
            
            r = self._residual_at_step(theta)
            while np.linalg.norm(r) / C.shape[0] > self.residual_threshold:
                C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)
                delta_theta = - np.linalg.pinv(C) @ r
                theta = theta + delta_theta.flatten()
                r = self._residual_at_step(theta)
            angle_history.append(theta.copy())
            ref_angle = theta[0]

            max_diff = np.max(np.abs(theta - self.theta_equil))
            if max_diff < self.max_diff_threshold:
                print(f"\nConverged: Max difference to target equilibrium ({max_diff:.2e}) is less than tolerance max_diff_threshold ({self.max_diff_threshold:.2e}).")
                break

        return angle_history[-1]
    
    def _find_equilibrium_and_export(self, ply_path=None, npy_path=None, gif_path=None, vtk_path = None, show_base=True):
        theta = np.zeros_like(self.theta_equil)
        c = self.init_step_size
        angle_history = []
        i = 0
        prev_ref_angle = theta[0]
        ref_angle = prev_ref_angle
        while c > self.step_size_threshold and i < self.MAX_ITERATION:
            i+=1
            C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)

            r = self._residual_at_step(theta)
            d = self.k * (theta - self.theta_equil)
            d = d[:, np.newaxis]

            C_rank = np.linalg.matrix_rank(C)

            if C_rank == C.shape[0]:
                H_1 = np.linalg.inv(self.H)
                G = H_1 @ C.T @ np.linalg.inv(C @ H_1 @ C.T)
                delta_theta = - (H_1 - G @ C @ H_1) @ d - G @ r
            else:
                dr = np.vstack((d, r))
                hcc0 = np.hstack((self.H, C.T))
                hcc0 = np.vstack((hcc0, np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))))
                delta_theta = - (np.linalg.pinv(hcc0) @ dr)[:len(theta), :]

            if i>2 and delta_theta[0]*(ref_angle - prev_ref_angle) < 0:
                c = c/2
            prev_ref_angle = ref_angle
            
            if np.max(np.abs(delta_theta)) > 0.0:
                theta = theta + delta_theta.flatten() * c / np.max(np.abs(delta_theta))
            
            r = self._residual_at_step(theta)
            while np.linalg.norm(r) / C.shape[0] > self.residual_threshold:
                C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)
                delta_theta = - np.linalg.pinv(C) @ r
                theta = theta + delta_theta.flatten()
                r = self._residual_at_step(theta)
            dof = self.num_creases - C_rank
            if dof == 0 : print(dof)
            
            angle_history.append(theta.copy())

            ref_angle = theta[0]

            max_diff = np.max(np.abs(theta - self.theta_equil))
            if max_diff < self.max_diff_threshold:
                print(f"\nConverged: Max difference to target equilibrium ({max_diff:.2e}) is less than tolerance max_diff_threshold ({self.max_diff_threshold:.2e}).")
                break

            if (i + 1) % (self.MAX_ITERATION // 100 if self.MAX_ITERATION >= 100 else 1) == 0 or i == self.MAX_ITERATION - 1:
                print(f"Processed iteration {i+1}/{self.MAX_ITERATION}, c = {c:.3e}, max_diff(deg) = {np.rad2deg(max_diff):.2f}", end='\r')        

        ah = np.vstack(angle_history)        # shape: (len(angle_history), n_creases)
        num_steps = ah.shape[0]

        # Compute arc-length (cumulative distance) along the curve
        delta = np.linalg.norm(np.diff(ah, axis=0), axis=1)  # Euclidean distance between steps
        arc_length = np.concatenate([[0], np.cumsum(delta)])  # total arc-length coordinate
        
        arc_uniform = np.linspace(0, arc_length[-1], self.num_interp_frames)

        print()
        print("Converged to equilibrium angles.")
        print(f"Processing {num_steps} iterations and interpolating to {self.num_interp_frames} frames.")
        # Interpolate each dimension over uniform arc-length
        ah_interp = np.zeros((self.num_interp_frames, ah.shape[1]))
        for i in range(ah.shape[1]):
            f = interp1d(arc_length, ah[:, i], kind='cubic')
            ah_interp[:, i] = f(arc_uniform)

        ah_raw        = ah
        ah_interp_raw = ah_interp

        print(f'Exporting results... | fps: {self.fps}, play_speed: {self.play_speed}')
        visualizer = OrigamiVisualizer(self.kirigami, ah_interp_raw, arc_uniform, show_base=show_base, ply_path=ply_path, npy_path=npy_path, face_colors=self.face_colors, reference_frame=self.reference_frame)
        visualizer.animate(fps=self.fps, play_speed=self.play_speed, gif_path=gif_path, vtk_path=vtk_path)
        print('Done.')

        return ah_raw

    def _find_equilibrium_trajectory(self):
        theta = self.theta_init
        if theta is None:
            theta = np.zeros_like(self.theta_equil)
        c = self.init_step_size
        angle_history = []
        i = 0
        prev_ref_angle = theta[0]
        ref_angle = prev_ref_angle
        while c > self.step_size_threshold and i < self.MAX_ITERATION:
            i+=1
            C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)

            r = self._residual_at_step(theta)
            d = self.k * (theta - self.theta_equil)
            d = d[:, np.newaxis]

            C_rank = np.linalg.matrix_rank(C)

            if C_rank == C.shape[0]:
                H_1 = np.linalg.inv(self.H)
                G = H_1 @ C.T @ np.linalg.inv(C @ H_1 @ C.T)
                delta_theta = - (H_1 - G @ C @ H_1) @ d - G @ r
            else:
                dr = np.vstack((d, r))
                hcc0 = np.hstack((self.H, C.T))
                hcc0 = np.vstack((hcc0, np.hstack((C, np.zeros((C.shape[0], C.shape[0]))))))
                delta_theta = - (np.linalg.pinv(hcc0) @ dr)[:len(theta), :]

            if i>2 and delta_theta[0]*(ref_angle - prev_ref_angle) < 0:
                c = c/2
            prev_ref_angle = ref_angle
            
            if np.max(np.abs(delta_theta)) > 0.0:
                theta = theta + delta_theta.flatten() * c / np.max(np.abs(delta_theta))
            
            r = self._residual_at_step(theta)

            k=0
            while np.linalg.norm(r) / C.shape[0] > self.residual_threshold:
                C = compute_constraint_matrix(self.closed_loop_home_screw, self.closed_loop_crease_idx, theta)
                delta_theta = - np.linalg.pinv(C) @ r
                theta = theta + delta_theta.flatten()
                r = self._residual_at_step(theta)
                k+=1
                if k>10000:
                    print(self.num_creases - C_rank)
            dof = self.num_creases - C_rank
            if dof == 0 : print(dof)
            
            angle_history.append(theta.copy())

            ref_angle = theta[0]

            max_diff = np.max(np.abs(theta - self.theta_equil))
            if max_diff < self.max_diff_threshold:
                print(f"\nConverged: Max difference to target equilibrium ({max_diff:.2e}) is less than tolerance max_diff_threshold ({self.max_diff_threshold:.2e}).")
                break

            if (i + 1) % (self.MAX_ITERATION // 100 if self.MAX_ITERATION >= 100 else 1) == 0 or i == self.MAX_ITERATION - 1:
                print(f"Processed iteration {i+1}/{self.MAX_ITERATION}, "
                        f"c = {c:.3e}, max_diff(deg) = {np.rad2deg(max_diff):.2f}, dof = {dof}", end='\r')        

        ah = np.vstack(angle_history)        # shape: (len(angle_history), n_creases)
        self.angle_history = ah
        print()
        print("Converged to equilibrium angles.")

        return ah
    
    def export_trajectory(self, angle_history=None, ply_path=None, npy_path=None, gif_path=None, vtk_path = None, show_base=True):
        if angle_history is None:
            angle_history = self.angle_history
        num_steps = angle_history.shape[0]
        # Compute arc-length (cumulative distance) along the curve
        delta = np.linalg.norm(np.diff(angle_history, axis=0), axis=1)  # Euclidean distance between steps
        arc_length = np.concatenate([[0], np.cumsum(delta)])  # total arc-length coordinate
        arc_uniform = np.linspace(0, arc_length[-1], self.num_interp_frames)

        
        print(f"Processing {num_steps} iterations and interpolating to {self.num_interp_frames} frames.")
        # Interpolate each dimension over uniform arc-length
        ah_interp = np.zeros((self.num_interp_frames, angle_history.shape[1]))
        for i in range(angle_history.shape[1]):
            f = interp1d(arc_length, angle_history[:, i], kind='cubic')
            ah_interp[:, i] = f(arc_uniform)

        print(f'Exporting results... | fps: {self.fps}, play_speed: {self.play_speed}')
        visualizer = OrigamiVisualizer(self.kirigami, ah_interp, arc_uniform, show_base=show_base, ply_path=ply_path, npy_path=npy_path, face_colors=self.face_colors, reference_frame=self.reference_frame)
        visualizer.animate(fps=self.fps, play_speed=self.play_speed, gif_path=gif_path, vtk_path=vtk_path)
        print('Done.')
        return

