import numpy as np
from time import process_time_ns
from tqdm import tqdm
from typing import List

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import copy

FaceKey = Tuple[int, int]  

"""
Final Rigid Foldable Structure Module
Author: GeonHee Cho
Last modified: 2026. 01. 19 ; hinging fix
"""

class RigidFoldableStructure:
    """
    Integrated Rigid-Foldable Structure Framework.
    Supports non-manifold complexes, localized screw loops, and detailed sequence visualization.
    """
    def __init__(
        self,
        vert: List[List[List[float]]],
        edge: List[List[List[int]]],
        face: List[List[List[int]]],
        connection: Optional[List[Tuple[FaceKey, FaceKey, str]]] = None,
        base: List[int] = [0,0],
    ):
        '''
        Initializes the structural parameters, orchestrates global orientation unification, 
        and constructs the initial kinematic graph and loop basis.
        '''
        self.verts_raw = vert
        self.edges_raw = [list(e) for e in edge]
        self.faces_raw = copy.deepcopy(face)
        self.connection = connection or []
        
        self.added = False
        for cc in self.connection:
            if cc[-1] == 'h':
                self.added = True
            
        self.conn_layer_idx = len(self.faces_raw) if self.connection else -1
        if self.added:
            self.edges_raw.append([]) 

        self._initialize_geometry()
        self.face_adj = self._build_graph_optimized()
        self._unify_global_orientations(base_face=(0, 0))
        
        self._update_flattened_faces()
        self._analyze_kinematic_loops(self.face_adj)
        
        self._finalize_graph_mapping(self.face_adj)
        self._build_input_mapping_table(self.face_adj)
        
        if connection is not None:
            self.print_new_hinges()
        else:
            print("[*] No inter-layer connections provided; skipping hinge metadata.")        
        self.sheet_sign(face)

    def sheet_sign(self, face_input: List[List[List[int]]]):
        '''
        Determines the orientation parity (1 or -1) of input facets relative to the internal 
        unified reference, accounting for cyclic permutations of vertex indices.
        '''
        self.sheet_signs = []
        for i, f_in in enumerate(face_input):
            a = self.faces_raw[i][0]
            b = f_in[0]
            try:
                idx = a.index(b[0])
                if a[(idx + 1) % len(a)] == b[1]:
                    self.sheet_signs.append(1)
                else:
                    self.sheet_signs.append(-1)
            except (ValueError, IndexError):
                self.sheet_signs.append(1)
                
        print(f"[*] Sheet Signs Generated: {self.sheet_signs}")     
                
    def get_final_desi(self, desi: List[List[float]]) -> List[List[float]]:
        '''
        Transforms the theoretical design parameters into localized folding angles 
        by applying the calculated orientation signs for topological consistency.
        '''
        if desi is None:
            return None
        
        if self.added:
            if len(desi) != len(self.sheet_signs) + 1:
                raise ValueError(f"Dimension mismatch: desi({len(desi)}) must be 1 greater than sheet_signs({len(self.sheet_signs)})")

            final_desi = [
                [val * self.sheet_signs[i] for val in layer]
                for i, layer in enumerate(desi[:-1])
            ]
            
            final_desi.append(desi[-1])
            
        else:
            final_desi = [
                [val * self.sheet_signs[i] for val in layer]
                for i, layer in enumerate(desi[:])
            ]

        return final_desi    
    
    def _unify_global_orientations(self, base_face: FaceKey):
        """ 
        Uses the pre-built face_adj (topological dual graph) to propagate 
        orientation consistency instead of raw coordinate matching.
        """
        visited = {base_face}
        queue = deque([base_face])
        
        while queue:
            u = queue.popleft()
            for v, conn_type in self.face_adj[u].items():
                if v not in visited:
                    shared_edge = self._get_shared_edge_topological(u, v)
                    if shared_edge:
                        if self._check_winding_inconsistency(u, v) and conn_type != "SOLDER" :
                            self.faces_raw[v[0]][v[1]].reverse()
                    
                    visited.add(v)
                    queue.append(v)
                    
    def _get_shared_edge_topological(self, u: FaceKey, v: FaceKey) -> Optional[Tuple[int, int]]:
            """
            Identifies the shared vertex indices between two connected facets.
            """
            f_u = set(self.faces_raw[u[0]][u[1]])
            f_v = set(self.faces_raw[v[0]][v[1]])
            shared = list(f_u.intersection(f_v))
            
            if len(shared) >= 2:
                return tuple(shared[:2]) 
            return None     
        
    def _check_winding_inconsistency(self, u_k: Tuple[int, int], v_k: Tuple[int, int]) -> bool:
        """
        Evaluates the geometric winding order consistency between two adjacent facets 
        using spatial coordinate matching instead of topological indices.
        """
        u_f = self.faces_raw[u_k[0]][u_k[1]]
        v_f = self.faces_raw[v_k[0]][v_k[1]]
        
        u_verts = [np.array(self.verts_raw[u_k[0]][idx]) for idx in u_f]
        v_verts = [np.array(self.verts_raw[v_k[0]][idx]) for idx in v_f]
        
        len_u = len(u_verts)
        len_v = len(v_verts)
        
        for i in range(len_u):
            p1_u = u_verts[i]
            p2_u = u_verts[(i + 1) % len_u]
            
            for j in range(len_v):
                p1_v = v_verts[j]
                p2_v = v_verts[(j + 1) % len_v]
                
                if np.allclose(p1_u, p1_v, atol=1e-6) and np.allclose(p2_u, p2_v, atol=1e-6):
                    return True
                
                if np.allclose(p1_u, p2_v, atol=1e-6) and np.allclose(p2_u, p1_v, atol=1e-6):
                    return False
                    
        return False

    
    def _initialize_geometry(self):
        '''
        Aggregates raw vertex data into a global coordinate space and pre-calculates 
        geometric properties such as offsets and centroids.
        '''
        self.vertex_offsets = np.cumsum([0] + [len(vs) for vs in self.verts_raw[:-1]])
        self.raw_vert_idx_to_global = {}
        cnt = 0
        for l, vs in enumerate(self.verts_raw):
            if not vs: continue
            for i in range(len(vs)):
                self.raw_vert_idx_to_global[(l, i)] = cnt
                cnt += 1
        flat_verts = [np.asarray(vs) for vs in self.verts_raw if vs]
        self.vertices = np.concatenate(flat_verts, axis=0) if flat_verts else np.empty((0,3))
        self.faces = [[v + self.vertex_offsets[l] for v in f] for l, f_l in enumerate(self.faces_raw) for f in f_l]

    def _update_flattened_faces(self):
        """Refreshes the global face index list after orientation unification."""
        self.faces = [[v + self.vertex_offsets[l] for v in f] 
                    for l, f_l in enumerate(self.faces_raw) for f in f_l]
        self.centroids = np.array([self.vertices[f].mean(axis=0) for f in self.faces])

    def _build_graph_optimized(self):
        '''
        Constructs an optimized dual graph representation, categorizing edges into 
        intra-layer, inter-layer hinges, and rigid soldered connections.
        '''
        graph, self.hinge_metadata = defaultdict(dict), []
        for l in range(len(self.faces_raw)):
            e_map = defaultdict(list)
            for f_idx, f_v in enumerate(self.faces_raw[l]):
                graph[(l, f_idx)] = {}
                for i in range(len(f_v)):
                    e_map[tuple(sorted((f_v[i], f_v[(i+1)%len(f_v)])))].append(f_idx)
            for f_list in e_map.values():
                for f1, f2 in combinations(f_list, 2):
                    u, v = (l, f1), (l, f2)
                    graph[u][v] = "INTRA"; graph[v][u] = "INTRA"

        for a, b, c_type in self.connection:
            u, v = tuple(a), tuple(b)
            if c_type == 'h':
                edge_info = self._link_hinge_connection(u, v, self.conn_layer_idx)
                if edge_info:
                    graph[u][v] = "INTER"; graph[v][u] = "INTER"
                    self.hinge_metadata.append({
                        'facets': (u, v), 
                        'dir': edge_info['dir'], 
                        'layer_p': edge_info['layer_p'], 
                        'theta_idx': (self.conn_layer_idx, edge_info['idx'])
                    })
            elif c_type == 's':
                graph[u][v] = "SOLDER"; graph[v][u] = "SOLDER"
        return graph

    def _link_hinge_connection(self, u, v, conn_l) -> Optional[Dict]:
        '''
        Identifies shared geometric boundaries between distinct layers to synthesize 
        new inter-layer hinge constraints.
        '''
        la, ia, lb, ib = *u, *v
        fa, fb = self.faces_raw[la][ia], self.faces_raw[lb][ib]
        for i in range(len(fa)):
            p1, p2 = fa[i], fa[(i+1)%len(fa)]
            v1, v2 = np.array(self.verts_raw[la][p1]), np.array(self.verts_raw[la][p2])
            for j in range(len(fb)):
                w1, w2 = np.array(self.verts_raw[lb][fb[j]]), np.array(self.verts_raw[lb][fb[(j+1)%len(fb)]])
                if (np.allclose(v1, w1) and np.allclose(v2, w2)) or (np.allclose(v1, w2) and np.allclose(v2, w1)):
                    new_idx = len(self.edges_raw[conn_l])
                    self.edges_raw[conn_l].append([p1, p2])
                    return {'idx': new_idx, 'dir': (p1, p2), 'layer_p': la}
        return None

    def _get_directed_edge(self, f_src: FaceKey, f_dst: FaceKey):
        '''
        Extracts the directed vertex pair representing the crease line between 
        two facets, oriented according to the source facet's winding order.
        '''
        l_s, i_s = f_src
        l_d, i_d = f_dst
        face_src = self.faces_raw[l_s][i_s]
        face_dst = self.faces_raw[l_d][i_d]
        dst_coords = [np.array(self.verts_raw[l_d][v]) for v in face_dst]
        
        for i in range(len(face_src)):
            p1_idx, p2_idx = face_src[i], face_src[(i+1)%len(face_src)]
            v1, v2 = np.array(self.verts_raw[l_s][p1_idx]), np.array(self.verts_raw[l_s][p2_idx])
            match_v1 = any(np.allclose(v1, dc) for dc in dst_coords)
            match_v2 = any(np.allclose(v2, dc) for dc in dst_coords)
            if match_v1 and match_v2:
                return (l_s, p1_idx, p2_idx)
        return (l_s, 0, 0)

    def _analyze_kinematic_loops(self, face_adj):
        '''
        Extracts the fundamental cycles (minimum cycle basis) of the assembly graph 
        to derive the governing kinematic loop equations.
        '''
        dual_g = nx.Graph()
        for u, nbrs in face_adj.items():
            for v in nbrs: dual_g.add_edge(u, v)
        
        self.crease_ids = sorted([
            tuple(sorted((u, v))) 
            for u, nbs in face_adj.items() 
            for v, t in nbs.items() 
            if u < v and t != "SOLDER"
        ])
        c2idx = {c: i for i, c in enumerate(self.crease_ids)}
        
        self.screw_loops, self.index_screws, self.loop_joints, self.loop_sequences = [], [], [], []

        for floop in nx.minimum_cycle_basis(dual_g):
            joints, creases = [], []
            for i in range(len(floop)):
                u, v = floop[i], floop[(i+1)%len(floop)]
                
                if face_adj[u][v] == "SOLDER":
                    continue
                    
                d_edge = self._get_directed_edge(u, v)
                joints.append(d_edge)
                creases.append((u, v))
            
            if not joints: continue 

            self.loop_sequences.append(floop)
            self.loop_joints.append(joints)
            self.screw_loops.append(self._compute_screw_matrix(joints, check=1))
            self.index_screws.append([c2idx[tuple(sorted(c))] for c in creases])

    def _compute_screw_matrix(self, joints: List[Tuple[int, int, int]], check: int) -> np.ndarray:
        '''
        Computes the screw axis matrix with Topological Concurrency Check.
        Uses vertex indices to determine if the loop is Spherical (SO(3)).
        '''
        S_list = []
        omegas = []
        
        for l, u, v in joints:
            p1 = np.array(self.verts_raw[l][u])
            p2 = np.array(self.verts_raw[l][v])
            vec = p2 - p1
            mag = np.linalg.norm(vec)
            
            w = vec / mag if mag > 1e-12 else np.zeros(3)
            omegas.append(w)
            
            if check:
                S_list.append(np.concatenate([w, np.cross(p1, w)]))
            else:
                S_list.append(w)

        is_concurrent = False
        if joints:
            l0, u0, v0 = joints[0]
            candidates = {u0, v0}
            
            for l_i, u_i, v_i in joints[1:]:
                if l_i != l0: 
                    break
                candidates &= {u_i, v_i}
                if not candidates:
                    break
            
            if candidates:
                is_concurrent = True

        if is_concurrent:
            return np.array(omegas).T
        else:
            return np.array(S_list).T    
        
    def _finalize_graph_mapping(self, face_adj: Dict):
        '''
        Establishes a rigorous indexing map between the topological dual graph 
        and the numerical crease-joint arrays.
        '''
        f_order = [(l, f) for l in range(len(self.faces_raw)) for f in range(len(self.faces_raw[l]))]
        f2i = {f: i for i, f in enumerate(f_order)}
        c2i = {tuple(sorted(c)): i for i, c in enumerate(self.crease_ids)}
        
        self.graph = {}
        for u, nbrs in face_adj.items():
            if u not in f2i: continue
            self.graph[f2i[u]] = {
                f2i[v]: c2i.get(tuple(sorted((u,v)))) 
                for v in nbrs if v in f2i
            }
            
        self.joint = np.array([
            [p1 + self.vertex_offsets[l], p2 + self.vertex_offsets[l]] 
            for u, v in self.crease_ids 
            for l, p1, p2 in [self._get_directed_edge(u, v)]
        ])

    def _build_input_mapping_table(self, face_adj):
        '''
        Constructs a bidirectional mapping table that links raw hierarchical inputs 
        to the internal flattened kinematic crease vector.
        '''
        self.input_to_crease = [{} for _ in range(len(self.edges_raw))]
        for m in self.hinge_metadata:
            tl, ti = m['theta_idx']
            crease_key = tuple(sorted(m['facets']))
            if crease_key in self.crease_ids:
                self.input_to_crease[tl][ti] = self.crease_ids.index(crease_key)

        for ci, (f1, f2) in enumerate(self.crease_ids):
            if face_adj[f1][f2] != "INTER":
                l, p1, p2 = self._get_directed_edge(f1, f2)
                for ei, ev in enumerate(self.edges_raw[l]):
                    if set(ev) == {p1, p2}:
                        self.input_to_crease[l][ei] = ci; break

    def get_internal_theta(self, theta_input: List[List[Optional[float]]]) -> np.ndarray:
        '''
        Translates a multi-layered list of folding angles into a standardized 1D array 
        suitable for numerical solvers and kinematic updates.
        '''
        if theta_input is None:
            return None
        if len(theta_input) != len(self.edges_raw): raise ValueError("Layer count mismatch.")
        internal_theta = np.zeros(len(self.crease_ids))
        for l, angles in enumerate(theta_input):
            if len(angles) != len(self.edges_raw[l]): raise ValueError(f"Layer {l} size mismatch.")
            for ei, angle in enumerate(angles):
                if angle is None or abs(float(angle)) < 1e-12: continue
                if ei in self.input_to_crease[l]: internal_theta[self.input_to_crease[l][ei]] = float(angle)
                elif l != self.conn_layer_idx: raise ValueError(f"[Boundary Violation] Layer {l}, Edge {ei} is Rigid.")
        return internal_theta

    def print_new_hinges(self):
        '''
        Displays a detailed diagnostic report of the inter-layer hinge metadata, 
        including their topological source and structural indices.
        '''
        if not self.hinge_metadata:
            return
        
        print(f"\n{'='*95}")
        print(f"{'Inter-layer Hinge Connection Metadata Summary':^95}")
        print(f"{'='*95}")
        
        header = f"{'Hinge Location':<18} | {'Facet A ':<15} | {'Facet B ':<15} | {'Structural Source (Layer: V -> V)':<30}"
        print(header)
        print(f"{'-'*95}")
        
        for m in self.hinge_metadata:
            tl, ti, fa, fb, u, v, lp = *m['theta_idx'], m['facets'][0], m['facets'][1], *m['dir'], m['layer_p']
            
            location = f"Layer {tl}, Idx {ti:<5}"
            facets_a = f"{str(fa):<15}"
            facets_b = f"{str(fb):<15}"
            source_info = f"Source-Layer {lp}: Vertex {u} -> {v}"
            
            print(f"{location} | {facets_a} | {facets_b} | {source_info}")
        
        print(f"{'='*95}\n")

    def visualize(self, filename: str = 'rfs_loops_detailed.png'):
        '''
        Generates a 3D visualization of the structure, highlighting the active 
        kinematic loops and their constituent screw axis vectors.
        '''
        num_loops = len(self.loop_joints)
        if num_loops == 0: 
            print("No kinematic loops found.")
            return
        
        fig = plt.figure(figsize=(7 * num_loops, 8))
        for i, joints in enumerate(self.loop_joints):
            ax = fig.add_subplot(1, num_loops, i + 1, projection='3d')
            ax.add_collection3d(Poly3DCollection([self.vertices[f] for f in self.faces], alpha=0.03, facecolors='gray'))
            
            for j_idx, (l, u, v) in enumerate(joints):
                p1, p2 = np.array(self.verts_raw[l][u]), np.array(self.verts_raw[l][v])
                vec = p2 - p1
                ax.quiver(p1[0], p1[1], p1[2], vec[0], vec[1], vec[2], color='blue', arrow_length_ratio=0.2, linewidth=2.5)
                mid_p = (p1 + p2) / 2
                ax.text(mid_p[0], mid_p[1], mid_p[2], f"J{j_idx}", color='red', fontsize=10, fontweight='bold')

            seq = self.loop_sequences[i]
            path_text = " → ".join([f"F{f}" for f in seq]) + f" → F{seq[0]}"
            ax.set_title(f"Kinematic Loop {i}\n{path_text}", fontsize=11, pad=20)
            
            joint_details = "\n".join([f"J{k}: L{l}, V{u}→V{v}" for k, (l, u, v) in enumerate(joints)])
            fig.text(0.5 / num_loops + i / num_loops, 0.05, f"[Joint Details]\n{joint_details}", 
                     ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

            v_all = self.vertices; mid, rng = (v_all.max(0)+v_all.min(0))/2, (v_all.max(0)-v_all.min(0)).max()/2
            ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def reset_info(self):
        '''
        Resets internal state variables (placeholder for stateful extensions).
        '''
        pass
    
    def stiff(self, stiffness: List[List[Optional[float]]]) -> np.ndarray:
        '''
        Maps hierarchical joint stiffness or weight parameters into the global 
        crease space for constrained optimization.
        '''
        if len(stiffness) != len(self.edges_raw): raise ValueError("Layer count mismatch.")
        internal_theta = np.zeros(len(self.crease_ids))
        for l, angles in enumerate(stiffness):
            if len(angles) != len(self.edges_raw[l]): raise ValueError(f"Layer {l} size mismatch.")
            for ei, angle in enumerate(angles):
                if angle is None or abs(float(angle)) < 1e-12: continue
                if ei in self.input_to_crease[l]: internal_theta[self.input_to_crease[l][ei]] = float(angle)
        return internal_theta

    def invert_final_desi(self, final_desi: List[List[float]]) -> List[List[float]]:
        '''
        Performs the inverse mapping of target angles, restoring the original 
        input representation by reapplying orientation parities.
        '''
        if final_desi is None:
            return None

        if self.added:
            if len(final_desi) != len(self.sheet_signs) + 1:
                raise ValueError(f"Dimension mismatch: input({len(final_desi)}) "
                                f"must be 1 greater than sheet_signs({len(self.sheet_signs)})")

            original_desi = [
                [val * self.sheet_signs[i] for val in layer]
                for i, layer in enumerate(final_desi[:-1])
            ]
            
            original_desi.append(final_desi[-1])
        else:
            original_desi = [
                [val * self.sheet_signs[i] for val in layer]
                for i, layer in enumerate(final_desi)
            ]

        return original_desi
    
    def invert_internal_theta(self, internal_theta: np.ndarray) -> List[List[Optional[float]]]:
        '''
        Scatters a flattened 1D state vector back into the original multi-layered 
        hierarchical data structure for external processing.
        '''
        if internal_theta is None:
            return None
        
        if len(internal_theta) != len(self.crease_ids):
            raise ValueError(f"Input size {len(internal_theta)} does not match "
                            f"crease count {len(self.crease_ids)}.")

        theta_output = []
        for layer_edges in self.edges_raw:
            theta_output.append([0.0] * len(layer_edges))

        for layer_idx, mapping_dict in enumerate(self.input_to_crease):
            for edge_idx, crease_idx in mapping_dict.items():
                theta_output[layer_idx][edge_idx] = float(internal_theta[crease_idx])

        return theta_output