# Gaussian-fPEPS

Gaussian Fermionic Tensor Network Toolkit

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行（默认d波配对）
python gfpeps_app.py

# 运行不同配对类型
python gfpeps_app.py --config-name=s_wave      # s波配对
python gfpeps_app.py --config-name=p_ip_wave   # p+ip波配对
python gfpeps_app.py --config-name=d_wave      # d波配对
```

## 支持的配对类型

- **d_wave**: d波配对（高Tc铜氧化物超导体）
- **s_wave**: s波配对（常规超导体）
- **p_ip_wave**: p+ip波配对（拓扑超导体）

## 依赖

- Python >= 3.8
- numpy
- pymanopt==2.0.0
- h5py
- bitarray
- hydra-core
- jax==0.4.26

## 配置

- `backend: gpu` - 使用GPU加速（推荐）
- `backend: cpu` - 使用CPU（兼容性好）
- `pairing_type` - 配对类型选择
