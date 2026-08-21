conda activate env_isaaclab
## 1. Real2Sim 环境
### Training num_envs=1024
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-RealSim-PegInsert-Direct-v0 --headless  --num_envs 1024 --checkpoint checkpoints/Factory.pth

### Data Collection and Test
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-RealSim-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint checkpoints/Factory.pth --headless

## 2. 齿轮装配
### Training
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --headless  --num_envs 1024

### Data Collection and Test
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-GearMesh-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_gear/test/nn/Factory.pth --headless
