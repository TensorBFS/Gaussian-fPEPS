# Pairing Types and Backend Configuration

## Pairing Types

The code now supports three different BCS pairing symmetries:

### 1. D-Wave Pairing (`d_wave`)
- **Physical System**: High-Tc cuprate superconductors
- **Pairing Function**: `Δ(k) = Δ_X * cos(kx) - Δ_Y * cos(ky)`
- **Properties**: Nodal structure, even function, conventional superconductor
- **Usage**: `pairing_type: "d_wave"`

### 2. S-Wave Pairing (`s_wave`)
- **Physical System**: Conventional superconductors (Al, Nb)
- **Pairing Function**: `Δ(k) = Δ_X * constant`
- **Properties**: Full gap, isotropic, conventional superconductor
- **Usage**: `pairing_type: "s_wave"`

### 3. P+IP Wave Pairing (`p_ip_wave`)
- **Physical System**: Topological superconductors, chiral p-wave
- **Pairing Function**: `Δ(k) = Δ_X * sin(kx) + i*Δ_Y * sin(ky)`
- **Properties**: Complex, chiral, topological, supports Majorana fermions
- **Usage**: `pairing_type: "p_ip_wave"`

## Configuration Examples

### D-Wave Configuration
```yaml
hamiltonian:
  ht: 1.0
  DeltaX: 0.5  # d-wave amplitude in x-direction
  DeltaY: 0.3  # d-wave amplitude in y-direction
  pairing_type: "d_wave"
```

### S-Wave Configuration
```yaml
hamiltonian:
  ht: 1.0
  DeltaX: 0.5  # s-wave amplitude (constant)
  DeltaY: 0.0  # not used for s-wave
  pairing_type: "s_wave"
```

### P+IP Wave Configuration
```yaml
hamiltonian:
  ht: 1.0
  DeltaX: 0.5  # p+ip amplitude in x-direction
  DeltaY: 0.3  # p+ip amplitude in y-direction
  pairing_type: "p_ip_wave"
```

## Backend Configuration

The `backend` parameter controls which device JAX uses for computation:

### GPU Backend
```yaml
backend: gpu
```
- Uses CUDA GPU acceleration
- Requires CUDA-compatible GPU
- Faster for large systems
- Used in SLURM environments with GPU allocation

### CPU Backend
```yaml
backend: cpu
```
- Uses CPU only
- Works on any system
- Slower but more compatible
- Good for testing and small systems

## Running Different Pairing Types

### Using Default Configuration
```bash
python gfpeps_app.py
```

### Using Specific Pairing Type
```bash
python gfpeps_app.py --config-name=d_wave
python gfpeps_app.py --config-name=s_wave
python gfpeps_app.py --config-name=p_ip_wave
```

### Using Custom Configuration
```bash
python gfpeps_app.py --config-name=my_config
```

## SLURM Examples

### D-Wave with GPU
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=D-Wave --pty python gfpeps_app.py --config-name=d_wave
```

### S-Wave with GPU
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=S-Wave --pty python gfpeps_app.py --config-name=s_wave
```

### P+IP Wave with GPU
```bash
srun --partition=gpu_h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G --time=24:00:00 --job-name=P-IP-Wave --pty python gfpeps_app.py --config-name=p_ip_wave
```

## Backend Verification

To verify that the backend is working correctly:

1. **Check JAX device**: The code will show which device JAX is using
2. **Monitor GPU usage**: Use `nvidia-smi` to monitor GPU utilization
3. **Performance**: GPU should be significantly faster than CPU for large systems

## Troubleshooting

### GPU Issues
- Ensure CUDA is properly installed
- Check GPU availability with `nvidia-smi`
- Try CPU backend if GPU is not available

### Import Issues
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path: The script automatically adds `src/` to Python path

### Configuration Issues
- Verify YAML syntax is correct
- Check that all required parameters are present
- Use default configuration as template 