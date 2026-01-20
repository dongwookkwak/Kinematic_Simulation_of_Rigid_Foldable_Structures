import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from math import sin, cos, tan, atan2, acos, pi
import pyvista as pv
import os

from modules.solver.origami import *
from modules.solver.exponential import *


##################################################
#
# Plot / Display some useful results of origami structure.
# (e.g. Crease pattern, Folding angle plot, 3D configuration, ...)
# Author: Jiook Chung
# Last modified: 2025.01.19
#
##################################################

epsilon = 1e-6

# Obtains the connectivity graph of the faces, connecting crease index written on edge of the graph
def get_graph(vertices, edges, faces):
    # Create a dictionary for quick edge lookup
    edge_dict = {tuple(sorted(edge)): idx for idx, edge in enumerate(edges)}
    graph = {i: {} for i in range(len(faces))}

    # Precompute edges for each face
    face_edges = [set(tuple(sorted((face[k], face[(k + 1) % len(face)]))) for k in range(len(face))) for face in faces]

    for i, edges1 in enumerate(face_edges):
        for j in range(i + 1, len(face_edges)):
            shared_edges = edges1 & face_edges[j]
            if len(shared_edges) == 1:
                edge_key = next(iter(shared_edges))
                edge_idx = edge_dict[edge_key]
                graph[i][j] = edge_idx
                graph[j][i] = edge_idx

    return graph



class OrigamiVisualizer:
    """
    Visualizer for folding progression using PyVista, with headless GIF export.
    Uses legacy get_coordinates to compute exact folded positions.
    Requires:
      - kirigami with vert_list, edge_list, face_list, graph
      - angle_series: (T, E) array of fold angles
    """
    def __init__(self, kirigami, angle_series, time_series, show_base=True, save_flag = True, ply_path=None, npy_path=None,face_colors = None, reference_frame = None):
        # Core data
        self.k = kirigami
        self.angles = np.asarray(angle_series)  # shape (T, E)
        self.time = np.asarray(time_series)  # shape (T,)

        self.base_idx = kirigami.raw_vert_idx_to_global[tuple(kirigami.base)] if hasattr(kirigami, 'base') else 0
        self.base_origin = kirigami.centroids[self.base_idx]

        # Flat vertices array (V,3)
        self.vertices = np.stack([v for v in self.k.vertices], axis=0)
        # Edge array (E,2)
        self.edges = np.array([e for e in self.k.joint], dtype=int)
        # Face list: list of lists of vertex indices
        self.faces = [f for f in self.k.faces]
        self.num_original_faces = len(self.faces)
        # Build placeholder mesh with flat state
        self.graph = self.k.graph
        self.ply_path = ply_path
        self.npy_path = npy_path

        self.reference_frame = reference_frame

        self.deformed_vertices = []

        self.faces, self.vertices, self.added_vertex_dictionary = self._augment_faces()
        self.num_augmented_faces = len(self.faces)

        if show_base:
            cells = np.hstack([[len(face)] + face for face in self.faces])
        else:
            cells = np.hstack([[len(face)] + face for i, face in enumerate(self.faces) if i != self.base_idx])
            self.num_original_faces -= 1
            self.num_augmented_faces -= 1
            if face_colors is not None:
                face_colors = face_colors.tolist()[:self.base_idx] + face_colors.tolist()[self.base_idx+1:]
                face_colors = np.array(face_colors)
 
        self.mesh = pv.PolyData(self.vertices.copy(), cells)

        if face_colors is None:
            face_colors = np.array([[200, 200, 200, 255] for _ in range(self.num_original_faces)])
        
        if len(face_colors) < self.num_augmented_faces:
            face_colors = face_colors.tolist() + [[200, 200, 200, 255]] * (self.num_augmented_faces - len(face_colors))
            face_colors = np.array(face_colors)
        elif len(face_colors) > self.num_augmented_faces:
            face_colors = face_colors[:self.num_augmented_faces]
        
        self.face_colors = face_colors
        self.mesh.cell_data["rgba"] = self.face_colors
        self.mesh.plot(scalars="rgba", rgba=True, preference="cell",use_transparency=False)

        if save_flag:
            # Save initial mesh to PLY file
            self.mesh.save(ply_path, texture='rgba')


    def _apply_step(self, step):
        """
        Compute folded 3D coordinates via legacy get_coordinates and update mesh.
        """
        # fold_angles expected in radians
        fold_angles = self.angles[step]
        if self.reference_frame is None:
            coords, _ = self.get_coordinates(
                vertices=self.vertices,
                edges=self.edges,
                faces=self.faces,
                graph=self.graph,
                fold_angles=fold_angles,
                base_idx = self.base_idx, origin = self.base_origin,
                added_vertex_dictionary=self.added_vertex_dictionary
            )
        else:
            coords, _ = self.get_coordinates_with_reference(
                vertices=self.vertices,
                edges=self.edges,
                faces=self.faces,
                graph=self.graph,
                fold_angles=fold_angles,
                reference_frame=self.reference_frame,
                added_vertex_dictionary=self.added_vertex_dictionary
            )
        # Update mesh points to folded coords
        self.mesh.points = coords
        self.deformed_vertices.append(coords.copy())

    def animate(self, fps=30, play_speed=0.25, gif_path=None, vtk_path = None):
        """
        Headless export of folding progression to a GIF file using exact folding kinematics.
        Does not open any GUI windows.
        """
        # Calculate the length of single frame in seconds
        single_frame = (1 / fps) * play_speed
        # Get the time marks to be used for each frame
        time_steps = np.arange(0, self.time[-1], single_frame)
        # Interpolate angles to match time steps
        self.angles = np.array([
            np.interp(time_steps, self.time, self.angles[:, i])
            for i in range(self.angles.shape[1])
        ]).T
        
        n_steps = self.angles.shape[0]
        # Prepare off-screen plotter
        if gif_path is not None:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(self.mesh, scalars='rgba', rgba=True, preference='cell', show_edges=True)
            plotter.open_gif(gif_path, fps=fps,palettesize=256)
        # Step through frames
        if vtk_path is not None:
            os.makedirs(vtk_path, exist_ok=True)

        for step in range(n_steps):
            self._apply_step(step)
            if gif_path is not None:
                plotter.write_frame()
            # --- export vtk per step ---
            if vtk_path is not None:
                coords = self.mesh.points
                out_path = os.path.join(vtk_path, f"step_{step:04d}.vtk")
                self._write_vtk_polydata_points_polys(coords, self.faces, out_path)
        if gif_path is not None:
            plotter.close()
        # Save deformed vertices to .npy file
        self.deformed_vertices = np.vstack(self.deformed_vertices)
        self.deformed_vertices = self.deformed_vertices.reshape(n_steps, len(self.vertices) * 3)
        if self.npy_path is not None:
            np.savetxt(self.npy_path, self.deformed_vertices)

    def _write_vtk_polydata_points_polys(self, points, faces, filepath):
        pts = np.asarray(points, dtype=float)
        num_points = pts.shape[0]
        polys = [list(map(int, f)) for f in faces]
        n_polys = len(polys)
        polys_size = sum(len(p) + 1 for p in polys)

        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Origami folding step\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {num_points} float\n")
            for p in pts:
                f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
            f.write(f"POLYGONS {n_polys} {polys_size}\n")
            for p in polys:
                f.write(str(len(p)) + " " + " ".join(str(int(i)) for i in p) + "\n")
    
    def _augment_faces(self):
        # Traverse the graph
        augmented_faces = self.faces.copy()
        augmented_vertices = self.vertices.copy().tolist()
        visited = [False for _ in range(len(self.faces))]
        queue = deque([self.base_idx])
        added_vertex_idx = len(self.vertices)

        # Keys are vertex indices, values are dictionaries with keys of projected face idx, values are the added vertex indices
        added_vertex_dictionary = {}

        while(queue):
            i = queue.popleft()
            if visited[i]:
                continue
            visited[i] = True
            # augmented_faces.append(self.faces[i])

            for j, e in self.graph[i].items():
                if e is not None or visited[j]:
                    if not visited[j]:
                        queue.append(j)
                    continue
                queue.append(j)

                # Calculate the normal vector of the face i
                v0 = self.vertices[self.faces[i][0]]
                v1 = self.vertices[self.faces[i][1]]
                v2 = self.vertices[self.faces[i][2]]
                normal_i = np.cross(v1 - v0, v2 - v0)
                normal_i /= np.linalg.norm(normal_i)

                # Calculate the normal vector of the face j
                u0 = self.vertices[self.faces[j][0]]
                u1 = self.vertices[self.faces[j][1]]
                u2 = self.vertices[self.faces[j][2]]
                normal_j = np.cross(u1 - u0, u2 - u0)
                normal_j /= np.linalg.norm(normal_j)

                for vert_idx in range(len(self.faces[i])):
                    if np.abs(np.dot(u0 - self.vertices[self.faces[i][vert_idx]], normal_j)) < epsilon or np.abs(np.dot(normal_i, normal_j)) < 1-epsilon:
                        continue  # Skip if the normals are parallel or intersection is not defined
                    
                    if vert_idx != len(self.faces[i]) - 1:
                        augmented_faces.append([self.faces[i][vert_idx], added_vertex_idx,
                                                added_vertex_idx + 1, self.faces[i][(vert_idx + 1) % len(self.faces[i])]])
                    else:
                        augmented_faces.append([self.faces[i][vert_idx], added_vertex_idx,
                                                added_vertex_idx - len(self.faces[i]) + 1, self.faces[i][(vert_idx + 1) % len(self.faces[i])]])
                    
                    # Calculate the offset of intersection point
                    offset = np.dot(u0 - self.vertices[self.faces[i][vert_idx]], normal_j) / np.dot(normal_i, normal_j)
                    intersection_point = self.vertices[self.faces[i][vert_idx]] + offset * normal_i

                    augmented_vertices.append(intersection_point)
                    if self.faces[i][vert_idx] not in added_vertex_dictionary:
                        added_vertex_dictionary[self.faces[i][vert_idx]] = {}
                    added_vertex_dictionary[self.faces[i][vert_idx]][j] = added_vertex_idx
                    
                    added_vertex_idx += 1

        augmented_vertices = np.array(augmented_vertices)
        return augmented_faces, augmented_vertices, added_vertex_dictionary
    
    def get_coordinates(self, vertices, edges, faces, graph, fold_angles = None, base_idx = None, origin = None, added_vertex_dictionary = None):
        if fold_angles is None:
            fold_angles = np.zeros(len(edges))
        
        if base_idx is None:
            base_idx = 0

        R = [np.identity(3) for _ in range(len(faces))]
        coords = np.array([[0.,0.,0.] for _ in range(len(vertices))])

        if origin is None:
            origin = vertices[faces[base_idx][0]]
        visited = [False for _ in range(len(faces))]
        fixed_coords = [False for _ in range(len(vertices))]
        for i in faces[base_idx]:
            coords[i] = vertices[i] - origin
            fixed_coords[i] = True
        queue = deque([base_idx])

        while(queue):
            i = queue.popleft()
            if visited[i]:
                continue
            visited[i] = True

            
            anchor = None
            for vv in faces[i]:
                if fixed_coords[vv]:
                    anchor = vv
                    break
            if anchor is None:
                anchor = faces[i][0]
            t_i = coords[anchor] - R[i] @ vertices[anchor]

            # Soldered connection
            for j, e in graph[i].items():
                if e is not None or visited[j]:
                    continue

                queue.append(j)
                R[j] = R[i].copy() 

                # coords = t_i + R[i] @ vertices
                for v in faces[j]:
                    coords[v] = t_i + R[j] @ vertices[v]
                    fixed_coords[v] = True
                
                # Calculate the normal vector of the face i
                v0 = coords[faces[i][0]]
                v1 = coords[faces[i][1]]
                v2 = coords[faces[i][2]]
                normal_i = np.cross(v1 - v0, v2 - v0)
                normal_i /= np.linalg.norm(normal_i)

                # Calculate the normal vector of the face j
                u0 = coords[faces[j][0]]
                u1 = coords[faces[j][1]]
                u2 = coords[faces[j][2]]
                normal_j = np.cross(u1 - u0, u2 - u0)
                normal_j /= np.linalg.norm(normal_j)

                for vert_idx in range(len(faces[i])):
                    # Check if the intersection point is in the dictionary
                    if faces[i][vert_idx] not in added_vertex_dictionary:
                        continue
                    elif j not in added_vertex_dictionary[faces[i][vert_idx]]:
                        continue
                    offset = np.dot(u0 - coords[faces[i][vert_idx]], normal_j) / np.dot(normal_i, normal_j)
                    intersection_point = coords[faces[i][vert_idx]] + offset * normal_i
                    added_vertex_idx = added_vertex_dictionary[faces[i][vert_idx]][j]
                    coords[added_vertex_idx] = intersection_point                

            # Hinged connections
            for j, e in graph[i].items():
                if e is None or visited[j]:
                    continue
                queue.append(j)

                u, v = edges[e]
                if faces[i].index(u) == (faces[i].index(v) + 1) % len(faces[i]):
                    u, v = v, u

                k = coords[v] - coords[u]
                nrm = np.linalg.norm(k)
                if nrm < 1e-12:
                    continue
                k = k / nrm

                R_temp = Exponential.matrix_exp3(Exponential.vec_to_so3(k * float(fold_angles[e])))
                R[j] = R_temp @ R[i]

                for w in faces[j]:
                    if not fixed_coords[w]:
                        coords[w] = coords[u] + R[j] @ (vertices[w] - vertices[u])
                        fixed_coords[w] = True
        return coords, R

    def get_coordinates_with_reference(self, vertices, edges, faces, graph, fold_angles=None, reference_frame=None, added_vertex_dictionary=None):
        """
        Reference frame given as a 3D vector consisting of 3 vertex indices.
        The first two vertices define the x-axis, and the third vertex defines the y-axis.
        The first vertex is used as fixed origin, other reference vertices are used to define the orientation.
        """
        coords, R = self.get_coordinates(vertices, edges, faces, graph, fold_angles=fold_angles, added_vertex_dictionary=added_vertex_dictionary)
        if isinstance(reference_frame[0], List):
            # convert to tuple
            for l in range(3):
                reference_frame[l] = tuple(reference_frame[l])
        idx0, idx1, idx2 = self.k.raw_vert_idx_to_global[reference_frame[0]], self.k.raw_vert_idx_to_global[reference_frame[1]], self.k.raw_vert_idx_to_global[reference_frame[2]]
        origin = coords[idx0]
        x_axis = coords[idx1] - origin
        x_axis /= np.linalg.norm(x_axis)
        y_axis = coords[idx2] - origin
        y_axis -= np.dot(y_axis, x_axis) * x_axis  # Remove x-component
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)  # Ensure orthogonality
        R_ref = np.column_stack((x_axis, y_axis, z_axis))
        coords = (R_ref.T @ (coords - origin).T).T  # Transform to reference frame
        R = [R_ref.T @ r for r in R]  # Transform rotation matrices to reference frame
        return coords, R
