 ## 安装流程
### 1. 安装 IsaacLab
参考 https:// isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
大致的安装过程如下：
(1) 先创建 isaac lab 的conda环境
(2) 激活 isaaclab 环境，用 pip 安装 isaacsim
(3) 最后安装 isaaclab

### 2. TacEx 安装
从gitee上克隆仓库：
```
git clone https://gitee.com/tang-peiyuan/isaac-force-manip.git
```
在项目根目录下运行：
```
conda activate env_isaaclab # 激活你的 isaaclab 虚拟环境
bash ./tacex.sh -i
```
安装其他额外的包
```
pip install ikpy
```

### 3. 安装过程可能遇到的问题
1. cmake 没安装，执行下面的命令安装cmake
```
conda install -c conda-forge cmake
```

### 4. 运行脚本
#### 4.1. 强化学习模型训练
```
# Peg-in-Hole 任务
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-v0 --headless

# 齿轮装配任务
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --headless  --num_envs 1024

# 拧螺母
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-v0 --headless  --num_envs 512
```

#### 4.2 使用训练好的强化学习模型采集数据
需要将下面脚本中的 <path/to/your/rl_model> 替换成训练好的强化学习模型的路径
```
# Peg-in-Hole 任务
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint <path/to/your/rl_model> --headless 

# 齿轮装配任务
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 1 --enable_cameras --checkpoint <path/to/your/rl_model> --headless 

# 拧螺母
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-NutThread-Direct-v0 --num_envs 1 --enable_cameras --checkpoint <path/to/your/rl_model> --headless 
```

#### 4.3. Lerobot 数据集格式转换
使用下面的脚本，将原始的数据转为Lerobot v2.0 格式，方便后续模型的训练
(1) 先转成 hdf5 格式
```
python scripts/to_hdf5.py --source_dir <path/to/your/data/folder> --output_dir </path/to/output/folder>
```
(2) hdf5 转为 Lerobot v2.0 格式
详细的过程参考 /home/zhuchengyang/dataset_convertor/README.md

#### 4.4. Pi0 模型推理
模型推理和采数据共用同一个脚本，但是需要做以下操作:
- 确认 pi0 服务器已经启动
- 修改 [factory_env_cfg.py](source/tacex_tasks/tacex_tasks/factory/factory_env_cfg.py#L251) 里面的 policy_cfg，将其从原来的 `None` 改成 `policy_cfg = PI0RemoteConfig()`

例如，使用 pi0 跑 peg-in-hole任务的脚本
```
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint <path/to/your/rl_model> --headless 
```