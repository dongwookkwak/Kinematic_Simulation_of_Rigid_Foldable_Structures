import numpy as np

"""
Exponential coordinate utilities for rigid-body kinematics and dynamics.
Author: Dongwook Kwak
Last modified: 2025. 04. 03
"""

class Exponential:
    @staticmethod
    def near_zero(scalar: float, tol: float = 1e-14) -> bool:
        """
        Checks if a scalar value is near zero within a small tolerance.

        Args:
            scalar (float): The scalar to check.
            tol (float): Tolerance threshold.

        Returns:
            bool: True if the scalar is near zero.
        """
        return abs(scalar) < tol
    
    @staticmethod
    def axis_ang3(expc3: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Converts a 3D exponential coordinate vector into a unit axis and angle.

        Args:
            expc3 (np.ndarray): A 3-element exponential coordinate vector.

        Returns:
            tuple: (omghat, theta)
                omghat (np.ndarray): Unit rotation axis (3D vector).
                theta (float): Rotation angle.
        """
        if expc3.shape != (3,):
            raise ValueError("Input must be a 3-element vector.")

        theta = np.linalg.norm(expc3)
        if theta == 0:
            raise ValueError("Zero vector provided. Axis is undefined.")

        omghat = expc3 / theta
        return omghat, theta

    @staticmethod
    def axis_ang6(expc6: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Converts a 6D exponential coordinate vector into a screw axis and angle.

        Args:
            expc6 (np.ndarray): A 6-element vector (S * theta).

        Returns:
            tuple: (S, theta)
                S (np.ndarray): Unit screw axis (6D vector).
                theta (float): Scalar magnitude.
        """
        if expc6.shape != (6,):
            raise ValueError("Input must be a 6-element vector.")

        theta = np.linalg.norm(expc6[0:3])
        if Exponential.near_zero(theta):
            theta = np.linalg.norm(expc6[3:6])

        if Exponential.near_zero(theta):
            raise ValueError("Zero twist vector provided. Screw axis is undefined.")

        S = expc6 / theta
        return S, theta

    @staticmethod
    def vec_to_so3(omega: np.ndarray) -> np.ndarray:
        if omega.shape != (3,):
            raise ValueError("Input must be a 3-element vector.")
        return np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
    
    @staticmethod
    def so3_to_vec(so3mat: np.ndarray) -> np.ndarray:
        """
        Converts a 3x3 skew-symmetric matrix in so(3) to a 3D angular velocity vector.

        Args:
            so3mat (np.ndarray): A 3x3 skew-symmetric matrix.

        Returns:
            np.ndarray: A 3-element vector.
        """
        if so3mat.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix.")
        
        return np.array([
            so3mat[2, 1],
            so3mat[0, 2],
            so3mat[1, 0]
        ])

    @staticmethod
    def vec_to_se3(V: np.ndarray) -> np.ndarray:
        """
        Converts a 6D spatial velocity vector into a 4x4 matrix in se(3).
        
        Args:
            V (np.ndarray): A 6-element vector [omega; v].

        Returns:
            np.ndarray: A 4x4 se(3) matrix.
        """
        if V.shape != (6,):
            raise ValueError("Input must be a 6-element vector.")
        
        omega = V[0:3]
        v = V[3:6]
        se3mat = np.zeros((4, 4))
        se3mat[0:3, 0:3] = Exponential.vec_to_so3(omega)
        se3mat[0:3, 3] = v
        return se3mat
    
    @staticmethod
    def se3_to_vec(se3mat: np.ndarray) -> np.ndarray:
        """
        Converts a 4x4 se(3) matrix into a 6D spatial velocity vector.

        Args:
            se3mat (np.ndarray): A 4x4 matrix in se(3).

        Returns:
            np.ndarray: A 6-element vector [omega; v].
        """
        if se3mat.shape != (4, 4):
            raise ValueError("Input must be a 4x4 matrix.")

        omega = np.array([
            se3mat[2, 1],
            se3mat[0, 2],
            se3mat[1, 0]
        ])
        v = se3mat[0:3, 3]
        return np.concatenate((omega, v))

    @staticmethod
    def trans_inv(T: np.ndarray) -> np.ndarray:
        """
        Computes the inverse of a homogeneous transformation matrix efficiently,
        using its special structure.

        Args:
            T (np.ndarray): A 4x4 transformation matrix.

        Returns:
            np.ndarray: The inverse of the transformation matrix.
        """
        if T.shape != (4, 4):
            raise ValueError("Input must be a 4x4 matrix.")

        R = T[0:3, 0:3]
        p = T[0:3, 3]
        invT = np.zeros((4, 4))
        invT[0:3, 0:3] = R.T
        invT[0:3, 3] = -R.T @ p
        invT[3, 3] = 1
        return invT

    @staticmethod
    def matrix_exp3(so3mat: np.ndarray) -> np.ndarray:
        """
        Computes the matrix exponential of a so(3) matrix to obtain a rotation matrix in SO(3).

        Args:
            so3mat (np.ndarray): A 3x3 skew-symmetric matrix.

        Returns:
            np.ndarray: A 3x3 rotation matrix in SO(3).
        """
        if so3mat.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix.")

        omgtheta = Exponential.so3_to_vec(so3mat)
        if Exponential.near_zero(np.linalg.norm(omgtheta)):
            return np.eye(3)

        omghat, theta = Exponential.axis_ang3(omgtheta)
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * (omgmat @ omgmat)

    @staticmethod
    def matrix_exp6(se3mat: np.ndarray) -> np.ndarray:
        """
        Computes the matrix exponential of a se(3) matrix to obtain a transformation matrix in SE(3).

        Args:
            se3mat (np.ndarray): A 4x4 matrix in se(3).

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix in SE(3).
        """
        if se3mat.shape != (4, 4):
            raise ValueError("Input must be a 4x4 matrix.")

        omgtheta = Exponential.so3_to_vec(se3mat[0:3, 0:3])
        if Exponential.near_zero(np.linalg.norm(omgtheta)):
            T = np.eye(4)
            T[0:3, 3] = se3mat[0:3, 3]
            return T
        else:
            omghat, theta = Exponential.axis_ang3(omgtheta)
            omgmat = se3mat[0:3, 0:3] / theta
            R = Exponential.matrix_exp3(se3mat[0:3, 0:3])
            V = (
                np.eye(3) * theta
                + (1 - np.cos(theta)) * omgmat
                + (theta - np.sin(theta)) * (omgmat @ omgmat)
            ) @ (se3mat[0:3, 3] / theta)

            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = V
            return T

    @staticmethod
    def fkin_space(M: np.ndarray, Slist: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
        """
        Computes forward kinematics in the space frame for an open chain robot.

        Args:
            M (np.ndarray): 4x4 home configuration of the end-effector.
            Slist (np.ndarray): 6xN matrix of screw axes in space frame.
            thetalist (np.ndarray): N-element vector of joint angles.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the end-effector frame.
        """
        if M.shape != (4, 4):
            raise ValueError("M must be a 4x4 matrix.")
        if Slist.shape[0] != 6 or Slist.shape[1] != thetalist.shape[0]:
            raise ValueError("Slist must be 6xN and thetalist must be N elements.")

        T = M.copy()
        for i in reversed(range(thetalist.shape[0])):
            screw_theta = Slist[:, i] * thetalist[i]
            T = Exponential.matrix_exp6(Exponential.vec_to_se3(screw_theta)) @ T

        return T
    
    @staticmethod
    def frot_space(R: np.ndarray, omega_list: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
        """
        Computes forward kinematics in the space frame for an open chain robot.

        Args:
            M (np.ndarray): 4x4 home configuration of the end-effector.
            Slist (np.ndarray): 6xN matrix of screw axes in space frame.
            thetalist (np.ndarray): N-element vector of joint angles.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the end-effector frame.
        """
        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 matrix.")
        if omega_list.shape[0] != 3 or omega_list.shape[1] != thetalist.shape[0]:
            raise ValueError("omega_list must be 3xN and thetalist must be N elements.")

        R_i = R.copy()
        for i in reversed(range(thetalist.shape[0])):
            omega_theta = omega_list[:, i] * thetalist[i]
            R_i = Exponential.matrix_exp3(Exponential.vec_to_so3(omega_theta)) @ R_i

        return R_i

    @staticmethod
    def ad(V: np.ndarray) -> np.ndarray:
        """
        Computes the Lie Bracket representation of a 6D spatial velocity (Lie bracket matrix).

        Args:
            V (np.ndarray): A 6-element spatial velocity vector.

        Returns:
            np.ndarray: A 6x6 ad matrix such that [ad_V] @ W = [V, W].
        """
        if V.shape != (6,):
            raise ValueError("Input must be a 6-element vector.")
        
        omega_hat = Exponential.vec_to_so3(V[0:3])
        v_hat = Exponential.vec_to_so3(V[3:6])
        return np.block([
            [omega_hat, np.zeros((3, 3))],
            [v_hat,     omega_hat]
        ])

    @staticmethod
    def adjoint(T: np.ndarray) -> np.ndarray:
        """
        Computes the 6x6 adjoint representation of a transformation matrix in SE(3).

        Args:
            T (np.ndarray): A 4x4 transformation matrix.

        Returns:
            np.ndarray: A 6x6 adjoint representation matrix.
        """
        if T.shape != (4, 4):
            raise ValueError("Input must be a 4x4 matrix.")
        
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        p_hat = Exponential.vec_to_so3(p)
        return np.block([
            [R,             np.zeros((3, 3))],
            [p_hat @ R,     R]
        ])

    @staticmethod
    def jacobian_space(Slist: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
        """
        Computes the space Jacobian for an open-chain manipulator.

        Args:
            Slist (np.ndarray): A 6xN matrix of screw axes.
            thetalist (np.ndarray): An N-element vector of joint angles.

        Returns:
            np.ndarray: A 6xN space Jacobian matrix.
        """
        n = thetalist.shape[0]
        if Slist.shape != (6, n):
            raise ValueError("Slist must be of shape (6, N) matching thetalist length.")
        
        Js = Slist.copy()
        T = np.eye(4)
        for i in range(1, n):
            screw_theta = Slist[:, i - 1] * thetalist[i - 1]
            T = T @ Exponential.matrix_exp6(Exponential.vec_to_se3(screw_theta))
            Js[:, i] = Exponential.adjoint(T) @ Slist[:, i]
        
        return Js
    
    @staticmethod
    def trunc_jacobian_space(omega_list: np.ndarray, thetalist: np.ndarray) -> np.ndarray:
        """
        Computes the truncated space Jacobian using only angular velocity axes (omega).
        Useful for origami systems where screw axes contain only 3D rotation parts.

        Args:
            omega_list (np.ndarray): A 3xN matrix of angular velocity vectors (omega).
            thetalist (np.ndarray): An N-element vector of joint angles.

        Returns:
            np.ndarray: A 3xN angular part of the space Jacobian.
        """
        n = thetalist.shape[0]
        if omega_list.shape != (3, n):
            raise ValueError("omega_list must be of shape (3, N) matching thetalist length.")

        Jw = omega_list.copy()  # Initial Jacobian (angular part only)
        R = np.eye(3)           # Initialize identity rotation

        for i in range(1, n):
            omega_theta = omega_list[:, i - 1] * thetalist[i - 1]
            R = R @ Exponential.matrix_exp3(Exponential.vec_to_so3(omega_theta))
            Jw[:, i] = R @ omega_list[:, i]

        return Jw




if __name__ == "__main__":
    # Basic test cases to verify implementation
    print("=== Testing vec_to_so3 ===")
    omega = np.array([1, 2, 3])
    print(Exponential.vec_to_so3(omega))

    print("\n=== Testing so3_to_vec ===")
    so3mat = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    print(Exponential.so3_to_vec(so3mat))

    print("\n=== Testing vec_to_se3 ===")
    V = np.array([1, 2, 3, 4, 5, 6])
    print(Exponential.vec_to_se3(V))

    print("\n=== Testing se3_to_vec ===")
    se3mat = np.array([[0, -3, 2, 4],
                       [3, 0, -1, 5],
                       [-2, 1, 0, 6],
                       [0, 0, 0, 0]])
    print(Exponential.se3_to_vec(se3mat))

    print("\n=== Testing matrix_exp3 ===")
    print(Exponential.matrix_exp3(so3mat))

    print("\n=== Testing matrix_exp6 ===")
    print(Exponential.matrix_exp6(se3mat))

    print("\n=== Testing fkin_space ===")
    M = np.array([[-1, 0, 0, 0],
                  [0, 1, 0, 6],
                  [0, 0, -1, 2],
                  [0, 0, 0, 1]])
    Slist = np.array([[0, 0,  1,  4, 0,    0],
                      [0, 0,  0,  0, 1,    0],
                      [0, 0, -1, -6, 0, -0.1]]).T
    thetalist = np.array([np.pi / 2, 3, np.pi])
    print(Exponential.fkin_space(M, Slist, thetalist))
