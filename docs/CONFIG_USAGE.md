# 配置文件使用说明

## 1. DeltaX和DeltaY的物理意义

### 物理上的一般情况：实数
- **常规超导体**：DeltaX和DeltaY是实数
- **配对强度**：表示电子对的配对强度
- **物理直觉**：实数参数更容易理解和调节

### 当前代码的实现
```python
# 输入：实数 DeltaX, DeltaY
# 输出：根据配对类型产生不同的配对振幅

# d-wave: 实数配对
Δ(k) = DeltaX * cos(kx) - DeltaY * cos(ky)

# s-wave: 实数配对  
Δ(k) = DeltaX (常数)

# p+ip: 复数配对（通过实数参数产生）
real = DeltaX * sin(kx) - DeltaY * sin(ky)
imag = DeltaX * sin(ky) + DeltaY * sin(kx)
Δ(k) = real + i*imag
```

### 物理合理性
✅ **这种设计是物理合理的**：
- 用户指定实数参数（物理直觉）
- 系统自动产生正确的配对函数
- 复数配对通过数学运算产生

## 2. 配置文件使用方法

### 方法1：使用预定义配置
```bash
# 使用默认配置
python gfpeps_app.py

# 使用特定配对类型
python gfpeps_app.py --config-name=d_wave
python gfpeps_app.py --config-name=s_wave
python gfpeps_app.py --config-name=p_ip_wave
```

### 方法2：使用自定义配置文件
```bash
# 指定自定义配置文件
python gfpeps_app.py --config-name=my_config

# 覆盖特定参数
python gfpeps_app.py --config-name=d_wave hamiltonian.DeltaX=0.8 hamiltonian.DeltaY=0.4
```

### 方法3：使用外部配置文件
```bash
# 使用外部YAML文件
python gfpeps_app.py --config-path=/path/to/configs --config-name=my_config
```

## 3. 创建自定义配置文件

### 在conf/gfpeps/目录下创建新文件
```yaml
# conf/gfpeps/my_config.yaml
params:
  Nv: 2
  seed: 123

lattice:
  Lx: 50
  Ly: 50

hamiltonian:
  ht: 1.0
  DeltaX: 0.6
  DeltaY: 0.4
  delta: 0.0
  Mu: 0.0
  solve_mu_from_delta: false
  pairing_type: "p_ip_wave"

file:
  LoadFile: "./data/my_config.h5"
  WriteFile: "./data/my_config.h5"
  SaveEachSteps: True

optimizer:
  MaxIter: 200
  gtol: 1E-8

backend: gpu
```

### 使用自定义配置
```bash
python gfpeps_app.py --config-name=my_config
```

## 4. 参数覆盖示例

### 覆盖单个参数
```bash
python gfpeps_app.py --config-name=d_wave hamiltonian.DeltaX=0.8
```

### 覆盖多个参数
```bash
python gfpeps_app.py --config-name=p_ip_wave \
  hamiltonian.DeltaX=0.7 \
  hamiltonian.DeltaY=0.5 \
  lattice.Lx=80 \
  lattice.Ly=80 \
  optimizer.MaxIter=150
```

### 覆盖嵌套参数
```bash
python gfpeps_app.py --config-name=s_wave \
  hamiltonian.pairing_type="d_wave" \
  hamiltonian.DeltaX=0.6 \
  hamiltonian.DeltaY=0.3
```

## 5. 配置文件结构

### 系统参数
```yaml
params:
  Nv: 2          # 虚拟费米子数量
  seed: 44       # 随机种子
```

### 晶格参数
```yaml
lattice:
  Lx: 101        # x方向系统大小
  Ly: 101        # y方向系统大小
```

### 哈密顿量参数
```yaml
hamiltonian:
  ht: 1.0                    # 跳跃振幅
  DeltaX: 0.5               # x方向配对振幅
  DeltaY: 0.3               # y方向配对振幅
  delta: 0.0                # 空穴密度
  Mu: 0.0                   # 化学势
  solve_mu_from_delta: false # 是否从空穴密度求解化学势
  pairing_type: "d_wave"     # 配对类型
```

### 文件配置
```yaml
file:
  LoadFile: "./data/default.h5"  # 输入文件
  WriteFile: "./data/default.h5"  # 输出文件
  SaveEachSteps: True             # 是否保存每步结果
```

### 优化参数
```yaml
optimizer:
  MaxIter: 100   # 最大迭代次数
  gtol: 1E-7     # 梯度容差
```

### 计算后端
```yaml
backend: gpu     # 计算后端 (gpu/cpu)
```

## 6. 最佳实践

### 配置文件命名
- 使用描述性名称：`d_wave_large.yaml`, `p_ip_small.yaml`
- 包含关键参数：`d_wave_Delta05.yaml`

### 参数组织
- 相关参数分组
- 添加注释说明
- 使用合理的默认值

### 版本控制
- 将配置文件加入版本控制
- 记录参数变化的原因
- 保存重要的配置组合 