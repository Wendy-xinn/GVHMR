class EgoDataset(Dataset):
    def __init__(self, root_dir, split_file, cfg, mode='train'):
        """
        root_dir: 数据集根目录
        split_file: data_splits.csv 路径
        mode: 'train' or 'val'
        """
        self.root_dir = Path(root_dir)
        self.split_file = split_file
        self.cfg = cfg

        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)
        self.img_size = cfg.MODEL.IMAGE_SIZE   # 目标尺寸224
        self.img_size_ori = cfg.MODEL.IMAGE_SIZE_ORI      # 原始长边1920
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std  = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)
        
        # 1. 从 CSV 获取序列名称
        df_split = pd.read_csv(self.split_file)
        if mode in df_split.columns:
            self.seq_names = df_split[mode].dropna().tolist()
        else:
            raise ValueError(f"Mode '{mode}' 不在 CSV 的列名中。可选列: {df_split.columns.tolist()}")
        
        # 定义子文件夹路径
        self.color_root = self.root_dir / "egocentric_color"
        self.calib_root = self.root_dir / "calibrations"
        self.smpl_root = self.root_dir / f"smpl_camera_wearer_{mode}"
        # self.seq_names = ["recording_20210907_S02_S01_01", "recording_20210921_S11_S10_01"]
        
        self.data_list = []
        self._prepare_data()

    def _prepare_data(self):
        """ 遍历所有序列，建立索引表 """
        for seq in self.seq_names:
            seq_path = self.color_root / seq
            if not seq_path.exists(): continue
            
            # 处理“不用管名称”的子文件夹 (通常是时间戳文件夹)
            sub_folders = [f for f in seq_path.iterdir() if f.is_dir()]
            if not sub_folders: continue
            seq_content_path = sub_folders[0] 
            
            # 读取 pv.txt 获取位姿和内参
            pv_txt_path = list(seq_content_path.glob("*_pv.txt"))[0]
            pv_info = self._parse_pv_txt(pv_txt_path)
            
            # 读取该序列的 calibration
            calib_path = self.calib_root / seq / "cal_trans" / "holo_to_kinect12.json"
            with open(calib_path, 'r') as f:
                holo_to_kinect = json.load(f) # 包含trans(四元数)
                T_h2m = np.array(holo_to_kinect['trans'])
                T_m2h = np.linalg.inv(T_h2m)
            
            # 遍历图像文件夹 PV
            img_dir = seq_content_path / "PV"
            for img_path in img_dir.glob("*.jpg"):
                img_name = img_path.stem 
                
                # 分割文件名获取 timestamp 和 frame_id
                parts = img_name.split('_', 1)
                if len(parts) < 2: continue
                timestamp, frame_id = parts[0], parts[1]
                # print(timestamp)
                # print(frame_id)
                
                # 匹配 SMPL 文件路径
                # 路径示例: RECORDING_NAME/body_idx_x/results/frame_xxxxx/000.pkl
                # 注意: 这里假设 body_idx_0 是 wearer，具体需根据数据确认
                smpl_search_pattern = f"{seq}/body_idx_*/results/{frame_id}/000.pkl"
                matching_pkls = list(self.smpl_root.glob(smpl_search_pattern))
                
                if matching_pkls and timestamp in pv_info:
                    self.data_list.append({
                        'img_path': str(img_path),
                        'pv_data': pv_info[timestamp],
                        'cx_cy': pv_info['meta'], # 这是个字典
                        'T_m2h': T_m2h,
                        'smpl_path': str(matching_pkls[0])
                    })

    def _parse_pv_txt(self, path):
        """ 解析 pv.txt 文件 """
        info = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            # 第一行: cx, cy, w, h
            meta = [float(x) for x in lines[0].strip().split(',')]
            info['meta'] = {'cx': meta[0], 'cy': meta[1]}
            
            # 后续行: timestamp, fx, fy, pv2world_transform
            for line in lines[1:]:
                data = line.strip().split(',')
                ts = data[0]
                fx, fy = float(data[1]), float(data[2])
                # 4x4 变换矩阵
                trans_mat = np.array([float(x) for x in data[3:]]).reshape(4, 4)
                T_w2c = np.linalg.inv(trans_mat)
                info[ts] = {'fx': fx, 'fy': fy, 'T_w2c': T_w2c}    # 这里的transform是pv camera坐标系到holo世界坐标系
        return info

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.__len__())
        item = self.data_list[idx]
        
        # 1. 读取图像(可以后面直接用图像的路径进行图像处理)
        # image = cv2.imread(item['img_path'])
        # 2. 读取 SMPL 参数，得到3d坐标
        with open(item['smpl_path'], 'rb') as f:
            smpl_data = pickle.load(f)
        # 需要从axis-angle转换成矩阵的形式
        # OpenCV (Y down, Z forward) -> OpenGL (Y up, Z back)  这里不对，训练应该用opencv的坐标系
        # flip_yz = torch.tensor([
        #     [1,  0,  0, 0],
        #     [0, -1,  0, 0],
        #     [0,  0, -1, 0],
        #     [0,  0,  0, 1]
        # ], dtype=torch.float32)
        T_w2c = torch.tensor(item['pv_data']['T_w2c'], dtype=torch.float32)
        T_m2h = torch.tensor(item['T_m2h'], dtype=torch.float32)
        # T_total = flip_yz @ T_w2c @ T_m2h  # (4, 4)
        T_total = T_w2c @ T_m2h  # (4, 4)

        global_orient = torch.tensor(smpl_data['global_orient'], dtype=torch.float32).view(-1, 3)
        global_orient = batch_rodrigues(global_orient)   # (B, 3, 3)

        R_m2c = T_total[:3, :3].clone().detach().float()
        global_orient_c = R_m2c @ global_orient
        r_obj = R.from_matrix(global_orient_c.numpy().squeeze())   # 转换成axis angle
        global_orient_camera = r_obj.as_rotvec().astype(np.float32)

        global_orient_ = global_orient.view(-1, 1, 3, 3)  # (B, 1, 3, 3)

        body_pose = torch.tensor(smpl_data['body_pose'], dtype=torch.float32).view(-1, 3)
        body_pose = batch_rodrigues(body_pose)           # (B*23, 3, 3)
        body_pose = body_pose.view(-1, 23, 3, 3)         # (B, 23, 3, 3)

        betas = torch.tensor(smpl_data['betas'], dtype=torch.float32).view(1, -1)
        transl = torch.tensor(smpl_data['transl'], dtype=torch.float32).view(1, 3)
        smpl_output = self.smpl(
            global_orient=global_orient_,
            body_pose=body_pose,
            betas=betas,
            transl = transl,
            pose2rot=False 
        )  # master坐标系下
        # joints_3d_master = smpl_output.joints.detach().cpu().numpy().squeeze(0)
        joints_3d_master = smpl_output.joints.squeeze(0)
        vertices_master = smpl_output.vertices
        # 扩展为齐次坐标 [B, 6890, 4]
        ones = torch.ones(vertices_master.shape[0], vertices_master.shape[1], 1).to(vertices_master.device)
        vertices_master_ = torch.cat([vertices_master, ones], dim=-1)
        vertices_cam = (T_total @ vertices_master_.transpose(1, 2)).transpose(1, 2).squeeze(0)[:, :3] # 截取回 [6890, 3]

        # print(joints_3d_master)
        # print(joints_np)
        # 手动构造齐次坐标: (N, 3) -> (N, 4)
        joints_homo = np.ones((joints_3d_master.shape[0], 4))
        joints_homo[:, :3] = joints_3d_master
        joints_3d_camera = (T_total @ joints_homo.T).T[:, :3] # (J, 3)
        joints_3d_np = joints_3d_camera.detach().cpu().numpy()
        # print(joints_3d_camera)

        # 3. 透视投影: Camera 3D -> Image 2D Pixel(取消2d的loss，因为z太小会导致u/v很大，没办法计算)
        fx, fy = item['pv_data']['fx'], item['pv_data']['fy']
        cx, cy = item['cx_cy']['cx'], item['cx_cy']['cy'] # 注意索引方式
        z = joints_3d_np[:, 2]
        u = (joints_3d_np[:, 0] * fx) / z + cx
        v = (joints_3d_np[:, 1] * fy) / z + cy
        # # print('u:', u)
        # # print('v:', v)
        
        # 组装为 (N, 3) 格式，最后一列是 visibility 标志位
        # 判定有效性
        # z > 0 是基础，0.01 是为了避开分母过小导致的数值不稳定
        valid_mask = (z > 0.1) 

        # 对于无效的点，把它们坐标设为 0 或者一个安全值，并将 visibility 设为 0
        u[~valid_mask] = 0
        v[~valid_mask] = 0

        # 组装时，最后一列是 visibility (1.0 代表有效，0.0 代表无效)
        keypoints_2d_input = np.stack([u, v, valid_mask.astype(np.float32)], axis=-1)
        J = joints_3d_np.shape[0]
        keypoints_3d_input = np.concatenate([joints_3d_np, np.ones((J, 1))], axis=-1)
        
        smpl_params = {'global_orient':  global_orient_camera,            #pose_np[:3],
                       'body_pose': smpl_data['body_pose'],
                       'betas': smpl_data['betas']
                      }
        has_smpl_params = {
            'global_orient': np.array([1.0], dtype=np.float32),
            'body_pose': np.array([1.0], dtype=np.float32),
            'betas': np.array([1.0], dtype=np.float32),
        }

        augm_config = self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            item['img_path'], item['cx_cy']['cx'], item['cx_cy']['cy'],
            self.img_size_ori, self.img_size_ori,
            keypoints_2d_input, keypoints_3d_input,
            smpl_params, has_smpl_params,
            self.flip_keypoint_permutation,
            self.img_size, self.img_size,
            self.mean, self.std,
            # self.train,
            False,
            augm_config
        )
        
            
        # 3. 准备返回数据
        sample = {
            'img': img_patch,
            "smpl_params": {
                "global_orient": torch.tensor(smpl_params['global_orient'], dtype=torch.float32),  # (3,)
                "body_pose": torch.tensor(smpl_params['body_pose'], dtype=torch.float32),          # (69,)
                "betas": torch.tensor(smpl_params['betas'], dtype=torch.float32)                   # (10,)
        },
            "keypoints_2d": torch.tensor(keypoints_2d, dtype=torch.float32),
            "keypoints_3d": torch.tensor(keypoints_3d, dtype=torch.float32),
            "has_smpl_params": { k: torch.tensor([1.0 if v else 0.0])
                                for k, v in has_smpl_params.items()},
            # "cx": torch.tensor(item['cx_cy']['cx'], dtype=torch.float32),
            # "cy": torch.tensor(item['cx_cy']['cy'], dtype=torch.float32)
            'vertices': vertices_cam.detach().cpu().numpy().astype(np.float32),
        }

        return sample