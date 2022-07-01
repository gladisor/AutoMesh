
import numpy as np
import open3d as o3d
import torch
#import pickle
#import os

from automesh.data.data import LeftAtriumData
from automesh.utils.utils import pad, is_mesh_file
from automesh.models.meshcnn.mesh import Mesh


class MeshCNNLeftAtriumData(LeftAtriumData):
    def __init__(self, mesh_paths, branch_paths, root: str, closest_k: int = 1, transform = None, pre_transform=None, pre_filter=None) -> None:
        super().__init__(mesh_paths, branch_paths, root, closest_k, transform, pre_transform, pre_filter)

        
    def get(self, idx):    
        #Points I need: 
        #1. Load Mesh and Segmentation labels
        mesh = o3d.io.read_triangle_mesh(self.mesh_paths[idx])
        branch = o3d.io.read_point_cloud(self.branch_paths[idx]) #check for new class with new branch type
        #2. pad label according to n_input_Edges
        #lines from mesh cnn #label = read_seg(sself.seg_paths[index]) - self.offset
        #label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
        #3. put files into Meta file
        meta = {}
        meta['mesh'] = mesh
        meta['label'] = branch
    
        #4. get and pad edge features
        edge_features = Mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        #5. store edge features in meta 
        meta['edge_features'] = edge_features
        ###(edge_features - self.mean) / self.std ##why this step??
        return meta
        
        
    def extract_features(self, mesh):
        features = []
        edge_points = self.get_edge_points(mesh)
        self.set_edge_lengths(mesh, edge_points)
        with np.errstate(divide='raise'):
            try:
                features.append(self.dihedral_angle(mesh, edge_points))
                features.append(self.symmetric_opposite_angles(mesh, edge_points))
                features.append(self.symmetric_ratios(mesh, edge_points))  
                return np.concatenate(features, axis=0)
            except Exception as e:
                print(e)
                raise ValueError(mesh.filename, 'bad features')
    
    def dihedral_angle(self,mesh, edge_points):
        normals_a = self.get_normals(mesh, edge_points, 0)
        normals_b = self.get_normals(mesh, edge_points, 3)
        dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
        angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
        return angles
    
    
    def symmetric_opposite_angles(self, mesh, edge_points):
        """ computes two angles: one for each face shared between the edge
            the angle is in each face opposite the edge
            sort handles order ambiguity
        """
        angles_a = self.get_opposite_angles(mesh, edge_points, 0)
        angles_b = self.get_opposite_angles(mesh, edge_points, 3)
        angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
        angles = np.sort(angles, axis=0)
        return angles
    
    
    def symmetric_ratios(self, mesh, edge_points):
        """ computes two ratios: one for each face shared between the edge
            the ratio is between the height / base (edge) of each triangle
            sort handles order ambiguity
        """
        ratios_a = self.get_ratios(mesh, edge_points, 0)
        ratios_b = self.get_ratios(mesh, edge_points, 3)
        ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
        return np.sort(ratios, axis=0)
    
    
    def get_edge_points(self, mesh):
        """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
            for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
            each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
        """
        edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
        for edge_id, edge in enumerate(mesh.edges):
            edge_points[edge_id] = self.get_side_points(mesh, edge_id)
            # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
        return edge_points
            
            
    def set_edge_lengths(self, mesh, edge_points=None):
         if edge_points is not None:
             edge_points = self.get_edge_points(mesh)
         edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
         mesh.edge_lengths = edge_lengths
   
    def get_normals(self, mesh, edge_points, side):
        edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
        edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
        normals = np.cross(edge_a, edge_b)
        div = self.fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
        normals /= div[:, np.newaxis]
        return normals
            
    def get_ratios(self, mesh, edge_points, side):
        edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                       ord=2, axis=1)
        point_o = mesh.vs[edge_points[:, side // 2 + 2]]
        point_a = mesh.vs[edge_points[:, side // 2]]
        point_b = mesh.vs[edge_points[:, 1 - side // 2]]
        line_ab = point_b - point_a
        projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / self.fixed_division(
            np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
        closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
        d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
        return d / edges_lengths  
    
    def get_side_points(mesh, edge_id):
        # if mesh.gemm_edges[edge_id, side] == -1:
        #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
        # else:
        edge_a = mesh.edges[edge_id]

        if mesh.gemm_edges[edge_id, 0] == -1:
            edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
            edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
        else:
            edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
            edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
        if mesh.gemm_edges[edge_id, 2] == -1:
            edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
            edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
        else:
            edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
            edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
        first_vertex = 0
        second_vertex = 0
        third_vertex = 0
        if edge_a[1] in edge_b:
            first_vertex = 1
        if edge_b[1] in edge_c:
            second_vertex = 1
        if edge_d[1] in edge_e:
            third_vertex = 1
        return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]

            
    def fixed_division(to_div, epsilon):
        if epsilon == 0:
            to_div[to_div == 0] = 0.1
        else:
            to_div += epsilon
        return to_div
        
        
        
        
        
        
        
        
        
# =============================================================================
#      self.opt = opt
#      self.mean = 0
#      self.std = 1
#      self.ninput_channels = None
#     ##for segmentation data class
#      # self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
#      self.root = opt.dataroot
#      self.dir = mesh_paths ###add appropriate root to mesh folder ->should contain train and test subfolders
#      self.classes, self.class_to_idx = self.find_classes(self.dir)
#      self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
#      self.nclasses = len(self.classes)
#      self.size = len(self.paths)
#      self.get_mean_std()
#      # modify for network later.
#      opt.nclasses = self.nclasses
#      opt.input_nc = self.ninput_channels
#             
#     def get_mean_std(self):
#         """ Computes Mean and Standard Deviation from Training Data
#         If mean/std file doesn't exist, will compute one:
#         returns
#         mean: N-dimensional mean
#         std: N-dimensional standard deviation
#         ninput_channels: N
#         (here N=5)
#         """
# 
#         mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
#         if not os.path.isfile(mean_std_cache):
#             print('computing mean std from train data...')
#             # doesn't run augmentation during m/std computation
#             num_aug = self.opt.num_aug
#             self.opt.num_aug = 1
#             mean, std = np.array(0), np.array(0)
#             for i, data in enumerate(self):
#                 if i % 20 == 0:
#                     print('{} of {}'.format(i, self.size))
#                 features = data['edge_features']
#                 mean = mean + features.mean(axis=1)
#                 std = std + features.std(axis=1)
#             mean = mean / (i + 1)
#             std = std / (i + 1)
#             transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
#                               'ninput_channels': len(mean)}
#             with open(mean_std_cache, 'wb') as f:
#                 pickle.dump(transform_dict, f)
#             print('saved: ', mean_std_cache)
#             self.opt.num_aug = num_aug
#         # open mean / std from file
#         with open(mean_std_cache, 'rb') as f:
#             transform_dict = pickle.load(f)
#             print('loaded mean / std from cache')
#             self.mean = transform_dict['mean']
#             self.std = transform_dict['std']
#             self.ninput_channels = transform_dict['ninput_channels']      
#             
#             
#             
#     def get(self, idx):
#         #Points I need: 
#         #1. Load Mesh and Segmentation labels
#         mesh = o3d.io.read_triangle_mesh(self.mesh_paths[idx])
#         branch = o3d.io.read_point_cloud(self.branch_paths[idx]) #check for new class with new branch type
#         #2. pad label according to n_input_Edges
#                 #lines from mesh cnn #label = read_seg(self.seg_paths[index]) - self.offset
#                 #label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
#         #3. put files into Meta file
#         meta = {}
#         meta['mesh'] = mesh
#         meta['label'] = branch
#     
#         #4. get and pad edge features
#         edge_features = Mesh.extract_features()
#         edge_features = pad(edge_features, self.opt.ninput_edges)
#         #5. store edge features in meta 
#         meta['edge_features'] = (edge_features - self.mean) / self.std
#         return meta
#         
# =============================================================================
    # from parent class torch_geometric.data.Dataset    
# =============================================================================
#     def get(self, idx):      
#         
#         ## loading from file
#         mesh = o3d.io.read_triangle_mesh(self.mesh_paths[idx])
#         branch = o3d.io.read_point_cloud(self.branch_paths[idx])
# 
#         ## extracting points
#         vertices = np.array(mesh.vertices)
#         orig_branch_points = np.array(branch.points)
#         _, branch_points_idx = np.unique(orig_branch_points, axis = 0, return_index = True)
#         branch_points = orig_branch_points[np.sort(branch_points_idx)]
# 
#         ## computing nearest neibhors on mesh
#         d = distance.cdist(branch_points, vertices)
#         mesh_branch_points = torch.tensor(d).topk(k = self.closest_k, dim = 1, largest = False).indices.squeeze(-1)
#         vertices = torch.tensor(vertices, dtype = torch.float32)
# 
#         data = Data(
#             x = vertices,
#             pos = vertices,
#             edge_index = LeftAtriumData.get_edges_from_triangles(np.array(mesh.triangles)),
#             y = mesh_branch_points)
# 
#         return data
#     
# =============================================================================
# ====================================================from segmentation_data.py
#     
#     def __getitem__(self, index):
#     
#         path = self.paths[index]
#         mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
#         meta = {}
#         meta['mesh'] = mesh
#         label = read_seg(self.seg_paths[index]) - self.offset
#         label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
#         meta['label'] = label
#         soft_label = read_sseg(self.sseg_paths[index])
#         meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
#         # get edge features
#         edge_features = mesh.extract_features()
#         edge_features = pad(edge_features, self.opt.ninput_edges)
#         meta['edge_features'] = (edge_features - self.mean) / self.std
#         return meta
#     
# =============================================================================
# ======================================================already in parent class 
#     def __len__(self):
#         return self.size
# =============================================================================

# ======================not needed because already implemented in parent class?
#     @staticmethod
#     def get_seg_files(paths, seg_dir, seg_ext='.seg'):
#         segs = []
#         for path in paths:
#             segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
#             assert(os.path.isfile(segfile))
#             segs.append(segfile)
#         return segs
# 
#     @staticmethod
#     def get_n_segs(classes_file, seg_files):
#         if not os.path.isfile(classes_file):
#             all_segs = np.array([], dtype='float64')
#             for seg in seg_files:
#                 all_segs = np.concatenate((all_segs, read_seg(seg)))
#             segnames = np.unique(all_segs)
#             np.savetxt(classes_file, segnames, fmt='%d')
#         classes = np.loadtxt(classes_file)
#         offset = classes[0]
#         classes = classes - offset
#         return classes, offset
# 
# =============================================================================
   
# =============================================================================
#     @staticmethod
#     def make_dataset(path):    #-------------------------------------------------->might need later
#         meshes = []
#         assert os.path.isdir(path), '%s is not a valid directory' % path
#     
#         for root, _, fnames in sorted(os.walk(path)):
#             for fname in fnames:
#                 if is_mesh_file(fname):
#                     path = os.path.join(root, fname)
#                     meshes.append(path)
#     
#         return meshes
#             
# =============================================================================
# =============================================================================
# import torch.utils.data as data
# import numpy as np
# import pickle
# import os
# 
# class BaseDataset(data.Dataset):  
#     def __init__(self, opt):
#         self.opt = opt
#         self.mean = 0
#         self.std = 1
#         self.ninput_channels = None
#         super(BaseDataset, self).__init__()
#         
# 
# def collate_fn(batch):
#     """Creates mini-batch tensors
#     We should build custom collate_fn rather than using default collate_fn
#     """
#     meta = {}
#     keys = batch[0].keys()
#     for key in keys:
#         meta.update({key: np.array([d[key] for d in batch])})
#     return meta
# ####
# 
# from models.layers.mesh import Mesh
# 
# class SegmentationData(BaseDataset):
# 
#     def __init__(self, opt):
#         BaseDataset.__init__(self, opt)
#         self.opt = opt
#         self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
#         self.root = opt.dataroot
#         self.dir = os.path.join(opt.dataroot, opt.phase)
#         self.paths = self.make_dataset(self.dir)
#         self.seg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg')
#         self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
#         self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
#         self.nclasses = len(self.classes)
#         self.size = len(self.paths)
#         self.get_mean_std()
#         # # modify for network later.
#         opt.nclasses = self.nclasses
#         opt.input_nc = self.ninput_channels
# 
#     def __getitem__(self, index):
#         path = self.paths[index]
#         mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
#         meta = {}
#         meta['mesh'] = mesh
#         label = read_seg(self.seg_paths[index]) - self.offset
#         label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
#         meta['label'] = label
#         soft_label = read_sseg(self.sseg_paths[index])
#         meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
#         # get edge features
#         edge_features = mesh.extract_features()
#         edge_features = pad(edge_features, self.opt.ninput_edges)
#         meta['edge_features'] = (edge_features - self.mean) / self.std
#         return meta
# 
#     def __len__(self):
#         return self.size
# 
#     @staticmethod
#     def get_seg_files(paths, seg_dir, seg_ext='.seg'):
#         segs = []
#         for path in paths:
#             segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
#             assert(os.path.isfile(segfile))
#             segs.append(segfile)
#         return segs
# 
#     @staticmethod
#     def get_n_segs(classes_file, seg_files):
#         if not os.path.isfile(classes_file):
#             all_segs = np.array([], dtype='float64')
#             for seg in seg_files:
#                 all_segs = np.concatenate((all_segs, read_seg(seg)))
#             segnames = np.unique(all_segs)
#             np.savetxt(classes_file, segnames, fmt='%d')
#         classes = np.loadtxt(classes_file)
#         offset = classes[0]
#         classes = classes - offset
#         return classes, offset
# 
#     @staticmethod
#     def make_dataset(path):
#         meshes = []
#         assert os.path.isdir(path), '%s is not a valid directory' % path
# 
#         for root, _, fnames in sorted(os.walk(path)):
#             for fname in fnames:
#                 if is_mesh_file(fname):
#                     path = os.path.join(root, fname)
#                     meshes.append(path)
# 
#         return meshes
# 
# 
# def read_seg(seg):
#     seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
#     return seg_labels
# 
# 
# def read_sseg(sseg_file):
#     sseg_labels = read_seg(sseg_file)
#     sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
#     return sseg_labels
# 
# 
# 
# 
# 
# =============================================================================
