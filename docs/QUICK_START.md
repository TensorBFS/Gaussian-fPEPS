# 快速开始指南

## 已完成的修改

### ✅ 1. 简化项目结构
- 删除了复杂的hatch配置
- 使用简单的`setup.py`和`requirements.txt`
- 修复了导入问题

### ✅ 2. 添加了三种配对类型
- **d_wave**: d波配对（高Tc铜氧化物超导体）
- **s_wave**: s波配对（常规超导体）  
- **p_ip_wave**: p+ip波配对（拓扑超导体）

### ✅ 3. 配置文件
- `conf/gfpeps/default.yaml` - 默认配置
- `conf/gfpeps/d_wave.yaml` - d波配置
- `conf/gfpeps/s_wave.yaml` - s波配置
- `conf/gfpeps/p_ip_wave.yaml` - p+ip波配置

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行不同配对类型

**默认d波配对：**
```bash
python gfpeps_app.py
```

**s波配对：**
```bash
python gfpeps_app.py --config-name=s_wave
```

**p+ip波配对：**
```bash
python gfpeps_app.py --config-name=p_ip_wave
```

**d波配对（明确指定）：**
```bash
python gfpeps_app.py --config-name=d_wave
```

### SLURM运行示例

**d波配对：**
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=D-Wave --pty python gfpeps_app.py --config-name=d_wave
```

**s波配对：**
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=S-Wave --pty python gfpeps_app.py --config-name=s_wave
```

**p+ip波配对：**
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=P-IP-Wave --pty python gfpeps_app.py --config-name=p_ip_wave
```

## Backend配置

### GPU后端（推荐）
```yaml
backend: gpu
```
- 使用CUDA GPU加速
- 需要CUDA兼容的GPU
- 大系统计算更快

### CPU后端
```yaml
backend: cpu
```
- 仅使用CPU
- 兼容性好
- 适合测试和小系统

## 配对类型说明

### D-Wave (d_wave)
- **物理系统**: 高Tc铜氧化物超导体
- **配对函数**: `Δ(k) = Δ_X * cos(kx) - Δ_Y * cos(ky)`
- **特性**: 节点结构，偶函数，常规超导体

### S-Wave (s_wave)
- **物理系统**: 常规超导体（Al, Nb）
- **配对函数**: `Δ(k) = Δ_X * constant`
- **特性**: 全能隙，各向同性，常规超导体

### P+IP Wave (p_ip_wave)
- **物理系统**: 拓扑超导体，手性p波
- **配对函数**: `Δ(k) = Δ_X * sin(kx) + i*Δ_Y * sin(ky)`
- **特性**: 复数，手性，拓扑，支持Majorana费米子

## 问题解决

### ModuleNotFoundError
- 脚本已自动修复，会自动添加`src/`到Python路径
- 确保运行`pip install -r requirements.txt`

### CUDA错误
- 这是环境问题，不影响代码功能
- 可以尝试使用CPU后端：`backend: cpu`

### 配置问题
- 检查YAML语法
- 使用提供的配置文件作为模板 