from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import terrain_utils

assert gymtorch

import torch

import numpy as np


class Terrain:
    def __init__(self, gym, sim, device, uneven=False):
        self.uneven = uneven
        self.gym = gym
        self.sim = sim
        self.device = device

        if uneven:
            self.build_uneven_terrain()
        else:
            self.build_ground_plane()

    def build_uneven_terrain(self):
        # a 40 * 40 terrain, resolution = 0.05m
        terrain_length = 80.0
        terrain_width = 80.0
        h_resolution = 0.05
        v_resolution = 0.002
        length_points = int(terrain_length // h_resolution)
        width_points = int(terrain_width // h_resolution)
        # border width: 1m
        border_width = 1.0
        border_point_len = int(border_width // h_resolution)
        heightfield = np.zeros((length_points, width_points), dtype=np.int16)

        # 19 waves (2.0m per wave)
        # height resolution: 0.002m
        # raise: 0.1m
        amplitude = 0.05 / v_resolution
        div_x = (length_points - 2 * border_point_len) / (19 * np.pi * 2)
        div_y = (width_points - 2 * border_point_len) / (19 * np.pi * 2)
        x = np.arange(border_point_len, length_points - border_point_len)
        y = np.arange(border_point_len, width_points - border_point_len)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(1, -1)
        wave_x = 0.5 * np.sin(xx / div_x)
        wave_y = 0.5 * np.sin(yy / div_y)
        heightfield[border_point_len : length_points - border_point_len, border_point_len : width_points - border_point_len] = (
            amplitude * wave_x * wave_y
        ).astype(heightfield.dtype)

        # heightfield causes error with 200+ envs, don't know why
        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(heightfield, h_resolution, v_resolution, 2.0)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -terrain_length / 2
        tm_params.transform.p.y = -terrain_width / 2
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = 0.0
        tm_params.dynamic_friction = 0.0
        tm_params.restitution = 1.0
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(order="C"), triangles.flatten(order="C"), tm_params)

        self.heightfield = torch.from_numpy(heightfield).float().to(self.device) * v_resolution
        self.h_resolution = h_resolution
        self.center_pos = torch.tensor((terrain_length / 2, terrain_width / 2)).to(self.device).unsqueeze(0)
        self.border_width = border_width

    def build_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

    def terrain_heights(self, horizontal_pos):
        query_len = len(horizontal_pos)
        if self.uneven:
            p = ((horizontal_pos + self.center_pos) / self.h_resolution).long().unsqueeze(0).repeat(4, 1, 1)
            p[[1, 3], :, 0] += 1
            p[[2, 3], :, 1] += 1
            px = torch.clip(p[:, :, 0], 0, self.heightfield.shape[0] - 1).flatten()
            py = torch.clip(p[:, :, 1], 0, self.heightfield.shape[1] - 1).flatten()
            sampled_height = self.heightfield[px, py].view(4, query_len).max(dim=0).values
            return sampled_height
        else:
            return torch.zeros(query_len, device=self.device)

    def local_max_heights(self, horizontal_pos):
        query_len = len(horizontal_pos)
        if self.uneven:
            p = ((horizontal_pos + self.center_pos) / self.h_resolution).long().unsqueeze(0).repeat(100, 1, 1)
            for i, x_offset in enumerate(range(-5, 5)):
                for j, y_offset in enumerate(range(-5, 5)):
                    p[i * 10 + j, :, 0] += x_offset
                    p[i * 10 + j, :, 1] += y_offset
            px = torch.clip(p[:, :, 0], 0, self.heightfield.shape[0] - 1).flatten()
            py = torch.clip(p[:, :, 1], 0, self.heightfield.shape[1] - 1).flatten()
            local_heights = self.heightfield[px, py].view(100, query_len)
            terrain_height = local_heights.max(dim=0).values
            return terrain_height
        else:
            return torch.zeros(query_len, device=self.device)
