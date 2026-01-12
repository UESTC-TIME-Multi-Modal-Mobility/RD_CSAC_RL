import torch
import einops
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
from pxr import Sdf, Gf
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni_drones.sensors.camera import Camera, PinholeCameraCfg
import dataclasses
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
import cv2
import os
import json

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        self.seed = cfg.seed
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        camera_cfg = PinholeCameraCfg(
            resolution=(224, 224),
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                horizontal_aperture=36.0,      # mm
                # vertical_aperture=36.0,        # mm
                focal_length=18.9,             # 取平均约为20mm
                clipping_range=(0.05, 5.0),    # 可视距离范围，按需求设置
            ),           
            sensor_tick=1,
            data_types=["distance_to_camera"],
            # 添加性能优化配置
            # colorize_semantic_segmentation=False,  # 禁用语义分割着色
            # semantic_filter_predicate="",  # 空过滤器
        )
        self.camera_sensor = Camera(camera_cfg)
        self.camera_sensor1 = Camera(camera_cfg)    

        # camera for visualization
        self.camera_sensor.initialize("/World/envs/env_.*/Camera0")  # 更新路径
        self.camera_sensor1.initialize("/World/envs/env_.*/Hummingbird_0/base_link/Camera1")  # 更新路径
        if cfg.headless == False:
            # 使用距离场可视化
            vis_camera_cfg = camera_cfg
            self.camera_vis = Camera(vis_camera_cfg)
            self.camera_vis.initialize("/World/envs/env_.*/Hummingbird_0/base_link/Camera")
            # 记录可视化相机是否需要更新
            self.has_vis_camera = True
        else:
            self.has_vis_camera = False
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.camera_images = None  #  先初始化为 None
        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            self.env_stats = {
                env_id: {
                    "velocity": [],       # 存储每步的速度
                    "acceleration": [],   # 存储每步的加速度
                    "orientation": []     # 存储每步的机身姿态
                }
                for env_id in range(self.num_envs)
            }
            # # 图像保存设置
            # self.frame_count = 0
            # self.save_images = True  # 设置为True来启用图像保存
            # self.image_save_dir = "camera_frames"
            # if self.save_images and not os.path.exists(self.image_save_dir):
            #     os.makedirs(self.image_save_dir)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        # ===== 创建摄像头 prim =====
        # 将相机创建在环境级别，而不是机体下面，这样可以避免机体旋转的影响
        for env_idx in range(self.num_envs):
            camera_prim_path = f"/World/envs/env_{env_idx}/Camera0"
            prim_utils.create_prim(
                prim_path=camera_prim_path,
                prim_type="Camera",
                translation=(0.0, 0.0, 2.0),
                orientation=[0.0, 0.0, 0.7071, 0.7071]
            )
        camera_prim_path1 = "/World/envs/env_0/Hummingbird_0/base_link/Camera1"  # 改为环境级别
        # camera_prim_path0 = "/World/envs/env_0/Camera0"  # 改为环境级别
        # camera_prim0 = prim_utils.create_prim(
        #     prim_path=camera_prim_path0,
        #     prim_type="Camera",
        #     translation=(0.0, 0.0, 2.0),  # 初始位置与无人机相同
        #     orientation=[0.0, 0.0, 0.7071, 0.7071]  # 恢复正确基准：天空在上，朝向Y+飞行方向
        # )
        camera_prim1 = prim_utils.create_prim(
            prim_path=camera_prim_path1,
            prim_type="Camera",
            translation=(0.0, 0.0, 0.0),  # 初始位置与无人机相同
            orientation=[0.5, 0.5, -0.5, -0.5]  # 恢复正确基准：天空在上，朝向Y+飞行方向
        )
        # print(f"[Scene] Created camera at {camera_prim_path0}")

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=self.cfg.seed,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                    'camera': UnboundedContinuousTensorSpec((1, 224, 224), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,)),
                "cost": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # cost Spec
        self.cost_spec = CompositeSpec({
            "agents": CompositeSpec({
                "cost": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)
        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "cost": UnboundedContinuousTensorSpec(1),  
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training ==True):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts
            
            # apply target pos
            self.target_pos[env_ids] = target_pos

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training == True):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        self.rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        self.rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(self.rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)

        # 更新相机位置和朝向：跟随无人机的位置和朝向
        self.update_camera_pose()
        # self.update_camera1_pose()
            # === 打印Camera0和无人机的yaw ===
        # env_idx = 0  # 只对第一个环境做对比，如需全部遍历可加循环
        # # 1. 获取Camera0四元数
        # camera0_pos, camera0_quat = self.get_camera0_pose(env_idx)
        # # 转为欧拉角
        # camera0_quat_xyzw = np.array([camera0_quat[1], camera0_quat[2], camera0_quat[3], camera0_quat[0]])  # [x, y, z, w]
        # camera0_euler = R.from_quat(camera0_quat_xyzw).as_euler('xyz')
        # camera0_yaw = camera0_euler[2]

        # # 2. 获取无人机四元数
        # drone_quat_xyzw = self.drone.rot[env_idx, 0].unsqueeze(0)
        # drone_euler = quaternion_to_euler(drone_quat_xyzw)  # [roll, pitch, yaw]
        # drone_yaw = drone_euler[0, 2].item()

        # print(f"[对比] Camera0 yaw: {camera0_yaw:.3f} rad, Drone yaw: {drone_yaw:.3f} rad, 差值: {abs(camera0_yaw-drone_yaw):.3f} rad")

        # camera0_pos, camera0_quat = self.get_camera0_pose()
        # print(f"Camera0 Position: {camera0_pos}, Quaternion: {camera0_quat}")
        self.camera_images = self.camera_sensor.get_images()["distance_to_camera"] # 获取摄像头图像
        
        # # 保存第一个环境的摄像头图像用于调试
        # if self.save_images and self.frame_count % 100 == 0:  # 每10帧保存一次
        #     self.save_camera_frame()
        # self.frame_count += 1

    
    def save_camera_frame(self):
        """保存摄像头距离场图像 - 为每个环境保存到不同位置"""
        if self.camera_images is not None:
            # 遍历所有环境，为每个环境保存图像
            for env_idx in range(min(10, self.num_envs)):  # 只保存前4个环境的图像
                # 创建每个环境的专用文件夹
                env_dir = f"{self.image_save_dir}/env_{env_idx}"
                if not os.path.exists(env_dir):
                    os.makedirs(env_dir)
                
                # 获取当前环境的图像数据
                img_data = self.camera_images[env_idx].cpu().numpy()  # 形状可能是 [H, W] 或 [C, H, W]
                
                # 处理图像维度 - 确保是2D图像
                if len(img_data.shape) == 3:  # [C, H, W]
                    if img_data.shape[0] == 1:  # 单通道
                        img_data = img_data[0]  # 转为 [H, W]
                    else:
                        img_data = img_data[0]  # 取第一个通道
                elif len(img_data.shape) != 2:
                    print(f"警告: 意外的图像维度 {img_data.shape}")
                    continue
                
                # 处理无效值：将NaN和inf转换为255
                img_data = np.where(np.isnan(img_data) | np.isinf(img_data), 255.0, img_data)
                
                # 归一化距离场数据到0-255范围
                # 先排除255值（原本的无效值）来计算正常值的范围
                valid_mask = img_data != 255.0
                if valid_mask.any() and img_data[valid_mask].max() > img_data[valid_mask].min():
                    # 对有效值进行归一化
                    valid_min = img_data[valid_mask].min()
                    valid_max = img_data[valid_mask].max()
                    img_normalized = np.zeros_like(img_data, dtype=np.uint8)
                    img_normalized[valid_mask] = ((img_data[valid_mask] - valid_min) / (valid_max - valid_min) * 254).astype(np.uint8)  # 使用0-254范围
                    img_normalized[~valid_mask] = 255  # 无效值设为255（白色）
                else:
                    # 如果没有有效值或所有有效值相等
                    img_normalized = np.full_like(img_data, 255, dtype=np.uint8)
                
                # 确保图像是正确的格式（2D灰度图）
                if len(img_normalized.shape) != 2:
                    print(f"警告: 归一化后图像维度错误 {img_normalized.shape}")
                    continue
                
                # 保存原始距离场数据（灰度图）
                filename = f"{env_dir}/depth_frame_{self.frame_count:06d}.png"
                try:
                    cv2.imwrite(filename, img_normalized)
                    
                    # 也保存彩色映射版本便于观察
                    img_colormap = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
                    filename_color = f"{env_dir}/depth_color_{self.frame_count:06d}.png"
                    cv2.imwrite(filename_color, img_colormap)
                    
                except Exception as e:
                    print(f"保存图像失败 (环境 {env_idx}): {e}")
                    print(f"图像形状: {img_normalized.shape}, 类型: {img_normalized.dtype}")
                    continue

            if self.frame_count % 10 == 0:  # 每10帧打印一次信息
                img_sample = self.camera_images[0].cpu().numpy()
                if len(img_sample.shape) == 3 and img_sample.shape[0] == 1:
                    img_sample = img_sample[0]
                print(f"保存图像帧 {self.frame_count}, 图像形状: {img_sample.shape}, 距离场范围: [{img_sample.min():.2f}, {img_sample.max():.2f}]")
    # def update_camera_pose(self):
    #     """更新相机位置和朝向：让相机始终与无人机机头方向一致"""
    #     try:
    #         for env_idx in range(self.num_envs):
    #             camera_prim_path = f"/World/envs/env_{env_idx}/Camera0"
    #             camera_prim = prim_utils.get_prim_at_path(camera_prim_path)
    #             if camera_prim is None:
    #                 continue

    #             # 获取无人机位置
    #             drone_pos_xyz = self.drone.pos[env_idx, 0].cpu().numpy().astype(np.float64)
    #             # 获取无人机四元数
    #             drone_quat_xyzw = self.drone.rot[env_idx, 0]
    #             # 机头方向（世界系，x轴）
    #             front_direction = quat_axis(drone_quat_xyzw.unsqueeze(0), axis=0).squeeze(0).cpu().numpy()
    #             front_direction = front_direction / np.linalg.norm(front_direction)
    #             # USD Camera本地 -Z 轴为前方，所以让 -Z 对准机头
    #             z_axis = -front_direction
    #             up = np.array([0, 0, 1], dtype=np.float32)
    #             x_axis = np.cross(up, z_axis)
    #             x_axis /= np.linalg.norm(x_axis)
    #             y_axis = np.cross(z_axis, x_axis)
    #             y_axis /= np.linalg.norm(y_axis)
    #             rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    #             quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
    #             quat_wxyz = np.roll(quat, 1)  # [w, x, y, z]

    #             # 设置相机位置和朝向
    #             camera_prim.GetAttribute("xformOp:translate").Set(tuple(drone_pos_xyz.tolist()))
    #             camera_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*quat_wxyz))
    #     except Exception as e:
    #         if hasattr(self, '_camera_update_error_count'):
    #             self._camera_update_error_count += 1
    #         else:
    #             self._camera_update_error_count = 1
    #         if self._camera_update_error_count <= 5:
    #             print(f"警告: 更新相机姿态失败 ({self._camera_update_error_count}/5): {e}")
    # def update_camera1_pose(self):
    #     """
    #     让 base_link 下的 Camera1 只跟随 yaw，不跟随 pitch/roll。
    #     原理：
    #     1. Camera1 会自动继承 base_link 的所有旋转
    #     2. 需要用 base_link 四元数的逆消除所有旋转
    #     3. 考虑 env 与 base_link 的坐标系误差
    #     4. 加回只含 yaw 的旋转
    #     5. 乘以基础 orientation 对齐相机坐标系
    #     """
    #     try:
    #         # Camera0 基础 orientation（世界坐标系 -> Camera0 局部）
    #         Q_env_cam0 = Gf.Quatd(0.0, 0.0, 0.7071, 0.7071)
    #         # Camera1 基础 orientation（base_link -> Camera1 局部）
    #         Q_baselink_cam1 = Gf.Quatd(0.5, 0.5, -0.5, -0.5)
    #         # env 与 base_link 的坐标系误差
    #         Q_env_baselink = Q_env_cam0 * Q_baselink_cam1.GetInverse()

    #         for env_idx in range(self.num_envs):
    #             camera_prim_path = f"/World/envs/env_{env_idx}/Hummingbird_0/base_link/Camera1"
    #             camera_prim = prim_utils.get_prim_at_path(camera_prim_path)
    #             if camera_prim is None:
    #                 continue

    #             # 获取 base_link 的四元数 [x, y, z, w]
    #             drone_quat_xyzw = self.drone.rot[env_idx, 0].cpu().numpy().astype(np.float64)
    #             # 转为 [w, x, y, z] 用于 Gf.Quatd
    #             drone_quat_wxyz = np.roll(drone_quat_xyzw, 1)
    #             drone_quat_gf = Gf.Quatd(
    #                 float(drone_quat_wxyz[0]), 
    #                 float(drone_quat_wxyz[1]), 
    #                 float(drone_quat_wxyz[2]), 
    #                 float(drone_quat_wxyz[3])
    #             )

    #             # 提取 yaw（世界坐标系下）
    #             drone_euler = quaternion_to_euler(torch.tensor(drone_quat_xyzw).unsqueeze(0))[0]
    #             yaw = float(drone_euler[2].item())

    #             # 只含 yaw 的四元数（世界 Z 轴）
    #             q_yaw = R.from_euler('z', yaw).as_quat()  # [x, y, z, w]
    #             q_yaw_wxyz = np.roll(q_yaw, 1)  # [w, x, y, z]
    #             q_yaw_gf = Gf.Quatd(
    #                 float(q_yaw_wxyz[0]), 
    #                 float(q_yaw_wxyz[1]), 
    #                 float(q_yaw_wxyz[2]), 
    #                 float(q_yaw_wxyz[3])
    #             )

    #             # 将只含 yaw 的四元数从世界系变换到 base_link 局部系
    #             q_yaw_baselink = Q_env_baselink.GetInverse() * q_yaw_gf * Q_env_baselink

    #             # Camera1 最终朝向 = base_link 四元数逆 * yaw（base_link 局部系）* 基础 orientation
    #             final_camera_quat = drone_quat_gf.GetInverse() * q_yaw_baselink * Q_baselink_cam1

    #             camera_prim.GetAttribute("xformOp:orient").Set(final_camera_quat)

    #     except Exception as e:
    #         print(f"Camera1 姿态更新失败: {e}")
    def update_camera_pose(self):
        """只用yaw，适配多环境，正确处理局部坐标系"""
        try:
            for env_idx in range(self.num_envs):
                camera_prim_path = f"/World/envs/env_{env_idx}/Camera0"
                camera_prim = prim_utils.get_prim_at_path(camera_prim_path)
                if camera_prim is None:
                    continue

                # 获取无人机世界位置
                drone_pos_world = self.drone.pos[env_idx, 0].cpu().numpy().astype(np.float64)
                
                # 获取 env 的原点位置（通过 env prim 的 translate 属性）
                env_prim_path = f"/World/envs/env_{env_idx}"
                env_prim = prim_utils.get_prim_at_path(env_prim_path)
                if env_prim is not None:
                    env_translate = env_prim.GetAttribute("xformOp:translate").Get()
                    if env_translate is not None:
                        env_origin = np.array(env_translate, dtype=np.float64)
                    else:
                        env_origin = np.zeros(3, dtype=np.float64)
                else:
                    env_origin = np.zeros(3, dtype=np.float64)
                
                # 计算相对于 env 局部坐标系的位置
                drone_pos_local = drone_pos_world - env_origin

                # 获取无人机四元数并提取 yaw
                drone_quat_xyzw = self.drone.rot[env_idx, 0].unsqueeze(0)
                drone_euler = quaternion_to_euler(drone_quat_xyzw)
                drone_yaw = drone_euler[0, 2].item()

                # 基础朝向为 Y+，加 90 度补偿
                base_camera_quat = Gf.Quatd(0.0, 0.0, 0.7071, 0.7071)
                yaw_offset = math.pi / 2
                yaw_rotation = Gf.Quatd(
                    math.cos((drone_yaw + yaw_offset) / 2.0), 
                    0.0, 
                    0.0, 
                    math.sin((drone_yaw + yaw_offset) / 2.0)
                )
                final_camera_quat = yaw_rotation * base_camera_quat

                # 设置相机局部位置和朝向
                camera_prim.GetAttribute("xformOp:translate").Set(tuple(drone_pos_local.tolist()))
                camera_prim.GetAttribute("xformOp:orient").Set(final_camera_quat)
                
        except Exception as e:
            if hasattr(self, '_camera_update_error_count'):
                self._camera_update_error_count += 1
            else:
                self._camera_update_error_count = 1
            if self._camera_update_error_count <= 5:
                print(f"警告: 更新相机姿态失败 ({self._camera_update_error_count}/5): {e}")
    def get_camera0_pose(self, env_idx=0):
        camera_prim_path = f"/World/envs/env_{env_idx}/Camera0"
        camera_prim = prim_utils.get_prim_at_path(camera_prim_path)
        if camera_prim is None:
            raise RuntimeError(f"Camera0 prim not found at {camera_prim_path}")

        # 读取位置
        pos = camera_prim.GetAttribute("xformOp:translate").Get()
        # 读取四元数（w, x, y, z）
        quat = camera_prim.GetAttribute("xformOp:orient").Get()
        # 返回为numpy格式
        pos_np = np.array(pos)
        quat_np = np.array([quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]])
        return pos_np, quat_np
    # def update_camera_pose(self):
    #     """更新相机位置和朝向：跟随无人机的位置和朝向"""
    #     try:
    #         # 为每个环境的主相机设置位置
    #         for env_idx in range(self.num_envs):
    #             camera_prim_path = f"/World/envs/env_{env_idx}/Camera0"  # 更新路径
                
    #             # 获取相机prim和无人机位置
    #             camera_prim = prim_utils.get_prim_at_path(camera_prim_path)
    #             if camera_prim is None:
    #                 continue
    #             # drone_pos_xyz = self.get_base_link_world_pose(env_idx=env_idx)[0]
    #             # 获取无人机的位置和偏航角
    #             drone_pos_xyz = self.drone.pos[env_idx, 0].cpu().numpy().astype(np.float64)
    #             # print(f"环境 {env_idx} 无人机位置: {drone_pos_xyz1}")   
    #             # 获取无人机的四元数并转换为欧拉角
    #             drone_quat_xyzw = self.drone.rot[env_idx, 0].unsqueeze(0)  # 保持tensor格式用于转换
                
    #             # 机头方向（世界系）
    #             front_direction = quat_axis(torch.tensor(drone_quat_xyzw).unsqueeze(0), axis=0).squeeze(0).cpu().numpy()
    #             front_direction = front_direction.reshape(3,)  # 保证是一维向量
    #             z_axis = -front_direction
    #             # 定义向上方向
    #             up = np.array([0, 0, 1], dtype=np.float32)

    #             # 构造旋转矩阵（摄像机z轴为up，y轴为前方）
    #             x_axis = np.cross(up, z_axis)
    #             x_axis /= np.linalg.norm(x_axis)
    #             y_axis = np.cross(z_axis, x_axis)
    #             y_axis /= np.linalg.norm(y_axis)
    #             rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    #             # 转为四元数（wxyz）
    #             quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
    #             quat_wxyz = np.roll(quat, 1)  # [w, x, y, z]
    #             # drone_euler = quaternion_to_euler(drone_quat_xyzw)  # 转换为欧拉角 [roll, pitch, yaw]
    #             # drone_yaw = drone_euler[0, 2].cpu().numpy() # 获取yaw角
    #                         # 只绕Z轴旋转（yaw），基础朝向为X+（机头方向）
    #             # camera_quat = Gf.Quatd(math.cos(drone_yaw/2.0), 0.0, 0.0, math.sin(drone_yaw/2.0))

    #             # # 基础相机朝向 (wxyz格式): 朝向Y+方向 - 不要修改这个
    #             # # base_camera_quat = Gf.Quatd(0.0, 0.0, 0.7071, 0.7071)  # [w, x, y, z] - 正确的基础朝向
    #             # base_camera_quat =  Gf.Quatd(1.0, 0.0, 0.0, 0.0) 
    #             # # 创建yaw旋转四元数 (绕Z轴旋转)
    #             # yaw_rotation = Gf.Quatd(math.cos(drone_yaw/2.0), 0.0, 0.0, math.sin(drone_yaw/2.0))
                
    #             # # 组合旋转：先应用基础朝向，再应用yaw旋转
    #             # final_camera_quat = base_camera_quat * yaw_rotation
    #             # final_camera_quat = yaw_rotation * base_camera_quat
    #             # 确保position和orientation属性存在
    #             translate_attr = camera_prim.GetAttribute("xformOp:translate")
    #             if not translate_attr:
    #                 camera_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
                
    #             orient_attr = camera_prim.GetAttribute("xformOp:orient")
    #             if not orient_attr:
    #                 camera_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)  # 使用双精度四元数

    #             # 更新xformOpOrder确保包含translate和orient
    #             xform_op_order_attr = camera_prim.GetAttribute("xformOpOrder")
    #             if not xform_op_order_attr:
    #                 camera_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
    #                 camera_prim.GetAttribute("xformOpOrder").Set(["xformOp:translate", "xformOp:orient"])
    #             else:
    #                 current_order = list(camera_prim.GetAttribute("xformOpOrder").Get())
    #                 if "xformOp:translate" not in current_order:
    #                     current_order.insert(0, "xformOp:translate")
    #                 if "xformOp:orient" not in current_order:
    #                     current_order.append("xformOp:orient")
    #                 camera_prim.GetAttribute("xformOpOrder").Set(current_order)
                
    #             # 更新位置和朝向
    #             camera_prim.GetAttribute("xformOp:translate").Set(tuple(drone_pos_xyz.tolist()))
                
    #             # 设置组合后的相机朝向
    #             camera_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*quat_wxyz))
            
    #     except Exception as e:
    #         # 如果出错，打印警告但不中断仿真
    #         if hasattr(self, '_camera_update_error_count'):
    #             self._camera_update_error_count += 1
    #         else:
    #             self._camera_update_error_count = 1
                
    #         # 只在前几次错误时打印，避免大量重复输出
    #         if self._camera_update_error_count <= 5:
    #             print(f"警告: 更新相机姿态失败 ({self._camera_update_error_count}/5): {e}")

    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame

        # Optional render for LiDAR and velocity visualization
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            # self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])
            drone_pos = self.drone.pos[0, 0]  # torch.Tensor, shape (3,)
            drone_vel = self.drone.vel_w[0, 0, :3]  # torch.Tensor, shape (3,)
            vel_scale = 2.0
            vel_end = drone_vel * vel_scale  # torch.Tensor, shape (3,)

            self.debug_draw.vector(
                drone_pos.unsqueeze(0),  # shape (1, 3)
                vel_end.unsqueeze(0),    # shape (1, 3)
                # color=[1.0, 0.0, 0.0],  # shape (1, 3)
            )
                        #     # 绘制目标方向向量（绿色）
            target_pos = self.target_pos[0, 0]
            direction_to_target = target_pos - drone_pos
            direction_normalized = direction_to_target / direction_to_target.norm().clamp(1e-6)
            direction_end = drone_pos + direction_normalized * 3.0  # 固定长度3米
            
            self.debug_draw.vector(
                drone_pos.unsqueeze(0),  # 起点
                (direction_end - drone_pos).unsqueeze(0),  # 向量
                color=(0.0, 1.0, 0.0,1.0),  # 绿色
                # thickness=2.0
            )
            # 绘制无人机朝向向量（蓝色）
            # 获取无人机的前向方向（机体坐标系的x轴方向）
            drone_quat = self.drone.rot[0, 0]  # [4] 四元数 (x,y,z,w)
            # 将机体x轴方向转换到世界坐标系
            body_x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            # 使用四元数旋转向量
            front_direction = quat_axis(drone_quat.unsqueeze(0), axis=0).squeeze(0)  # 机体x轴旋转到世界坐标系
            front_end = drone_pos + front_direction * 2.0  # 固定长度2米
            
            self.debug_draw.vector(
                drone_pos.unsqueeze(0),  # 起点
                (front_end - drone_pos).unsqueeze(0),  # 向量
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                # thickness=2.0
            )
            drone_pos1 = self.drone.pos[0, 0].clone()
            drone_pos1[2] -= 0.1  # z轴降低10cm
            self.debug_draw.vector(
                drone_pos1.unsqueeze(0),  # 起点
                (front_end - drone_pos).unsqueeze(0),  # 向量
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                # thickness=2.0
            )
            drone_pos2 = self.drone.pos[0, 0].clone()
            drone_pos2[2] -= 0.2  # z轴降低20cm
            self.debug_draw.vector(
                drone_pos.unsqueeze(0),  # 起点
                (front_end - drone_pos).unsqueeze(0),  # 向量
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                # thickness=2.0
            )
            drone_pos3 = self.drone.pos[0, 0].clone()
            drone_pos3[2] += 0.1  # z轴降低30cm
            self.debug_draw.vector(
                drone_pos3.unsqueeze(0),  # 起点
                (front_end - drone_pos).unsqueeze(0),  # 向量
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                # thickness=2.0
            )
            drone_pos4 = self.drone.pos[0, 0].clone()
            drone_pos4[2] += 0.2  # z轴降低40cm
            self.debug_draw.vector(
                drone_pos4.unsqueeze(0),  # 起点
                (front_end - drone_pos).unsqueeze(0),  # 向量
                color=(1.0, 0.0, 0.0, 1.0),  # 红色
                # thickness=2.0
                    )
            # 获取Camera0的位置和四元数
            camera0_pos, camera0_quat = self.get_camera0_pose(env_idx=0)
            # Camera0四元数 [w, x, y, z] -> [x, y, z, w] for scipy
            camera0_quat_xyzw = np.array([camera0_quat[1], camera0_quat[2], camera0_quat[3], camera0_quat[0]])
            camera0_rot = R.from_quat(camera0_quat_xyzw)
            camera0_forward = camera0_rot.apply([0, 0, -1])  # 用-Z方向
            camera0_forward = camera0_forward / np.linalg.norm(camera0_forward)
            camera0_forward_end = camera0_pos + camera0_forward * 2.0

            self.debug_draw.vector(
                torch.tensor(camera0_pos).unsqueeze(0),
                torch.tensor(camera0_forward_end - camera0_pos).unsqueeze(0),
                color=(0.0, 1.0, 1.0, 1.0),
            )

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states,
            "camera": self.camera_images if self.camera_images is not None else torch.zeros(self.num_envs, 1, 224, 224, device=self.device)
        }

        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))
        

        # b. safety reward for dynamic obstacles
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]


        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision | dynamic_collision
        
        # Final reward calculation
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = reward_vel + 1. + reward_safety_static * 1.0 + reward_safety_dynamic * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0
        else:
            self.reward = reward_vel + 1. + reward_safety_static * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0

        # cost
        # cost_safety_static = 
        self.cost = collision.float()
        # self.reward[collision] -= 50. # collision

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        self.terminated = below_bound | above_bound | collision
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()
        self.stats["cost"] += self.cost

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
            
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        cost = self.cost
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                    "cost" : cost
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
