# PeRFlow Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for the PeRFlow (Piecewise Rectified Flow) scheduler in the diffusers library. The framework has been established with all classes, methods, and comprehensive tests. This plan outlines the step-by-step approach to implement the actual functionality.

## Current Status

### Framework Completed ✓

All framework components are in place with proper structure and comprehensive test coverage:

- **Source Files**: 564 lines across 3 files
- **Test Suite**: 1,251 lines with 69 test methods
- **Documentation**: Complete API documentation and test specifications
- **Registration**: Fully integrated into diffusers package exports

### Implementation Status

All methods currently raise `NotImplementedError`. This document provides the roadmap to implement each component.

---

## Implementation Roadmap

### Phase 1: Core Scheduler Implementation

#### 1.1 TimeWindows Class

**File**: `diffusers/src/diffusers/schedulers/scheduling_perflow.py`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 29-60

**Implementation Steps**:

1. **`__init__(self, t_initial=1, t_terminal=0, num_windows=4, precision=1./1000)`**
   - Validate: `t_terminal < t_initial`
   - Calculate window boundaries:
     ```python
     time_windows = [1.*i/num_windows for i in range(1, num_windows+1)][::-1]
     self.window_starts = time_windows  # e.g., [1.0, 0.75, 0.5, 0.25]
     self.window_ends = time_windows[1:] + [t_terminal]  # e.g., [0.75, 0.5, 0.25, 0]
     self.precision = precision
     ```

2. **`get_window(self, tp: float) -> Tuple[float, float]`**
   - Find the window containing timepoint `tp`
   - Use precision offset for numerical stability: `(tp - 0.1*self.precision)`
   - Return `(window_start, window_end)`
   - **Edge case**: Handle boundary values with precision tolerance

3. **`lookup_window(self, timepoint: torch.FloatTensor)`**
   - Handle scalar tensors (dim == 0)
   - Handle batched tensors (batch iteration)
   - Return `(t_start, t_end)` tensors with same shape as input
   - **Performance**: Consider vectorization for large batches

**Tests to Pass**: `test_window_calculation`, `test_get_window_alpha`

**Success Criteria**: All 30 scheduler tests pass after TimeWindows implementation

---

#### 1.2 Beta Schedule Generation

**Function**: `betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine")`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 81-123

**Implementation Steps**:

1. Define alpha_bar functions:
   - **Cosine**: `math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2`
   - **Exp**: `math.exp(t * -12.0)`
   - Raise `ValueError` for unsupported types

2. Calculate betas:
   ```python
   betas = []
   for i in range(num_diffusion_timesteps):
       t1 = i / num_diffusion_timesteps
       t2 = (i + 1) / num_diffusion_timesteps
       betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
   return torch.tensor(betas, dtype=torch.float32)
   ```

**Tests to Pass**: `test_schedules`, `test_betas`

---

#### 1.3 PeRFlowScheduler Initialization

**Method**: `__init__(...)`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 156-200

**Implementation Steps**:

1. **Beta schedule setup**:
   - If `trained_betas` provided: `self.betas = torch.tensor(trained_betas, dtype=torch.float32)`
   - Elif `beta_schedule == "linear"`: `torch.linspace(beta_start, beta_end, num_train_timesteps)`
   - Elif `beta_schedule == "scaled_linear"`: `torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2`
   - Elif `beta_schedule == "squaredcos_cap_v2"`: Use `betas_for_alpha_bar()`
   - Else: Raise `NotImplementedError`

2. **Alpha computation**:
   ```python
   self.alphas = 1.0 - self.betas
   self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
   ```

3. **Final alpha setup**:
   ```python
   self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
   ```

4. **Initialize components**:
   ```python
   self.init_noise_sigma = 1.0
   self.time_windows = TimeWindows(t_initial=t_noise, t_terminal=t_clean, 
                                    num_windows=num_time_windows, 
                                    precision=1./num_train_timesteps)
   ```

5. **Validate prediction_type**:
   ```python
   assert prediction_type in ["ddim_eps", "diff_eps", "velocity"]
   ```

**Tests to Pass**: `test_timesteps`, `test_betas`, `test_schedules`, `test_prediction_type`

---

#### 1.4 Timestep Scheduling

**Method**: `set_timesteps(self, num_inference_steps: int, device=None)`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 220-248

**Implementation Steps**:

1. **Minimum steps validation**:
   ```python
   if num_inference_steps < self.config.num_time_windows:
       num_inference_steps = self.config.num_time_windows
       print(f"### num_inference_steps set as {self.config.num_time_windows}")
   ```

2. **Distribute steps across windows**:
   ```python
   timesteps = []
   for i in range(self.config.num_time_windows):
       if i < num_inference_steps % self.config.num_time_windows:
           num_steps_cur_win = num_inference_steps // self.config.num_time_windows + 1
       else:
           num_steps_cur_win = num_inference_steps // self.config.num_time_windows
       
       t_s = self.time_windows.window_starts[i]
       t_e = self.time_windows.window_ends[i]
       timesteps_cur_win = np.linspace(t_s, t_e, num=num_steps_cur_win, endpoint=False)
       timesteps.append(timesteps_cur_win)
   ```

3. **Convert to integer timesteps**:
   ```python
   timesteps = np.concatenate(timesteps)
   self.timesteps = torch.from_numpy(
       (timesteps * self.config.num_train_timesteps).astype(np.int64)
   ).to(device)
   ```

**Tests to Pass**: `test_timesteps_generation`, `test_minimum_inference_steps`, `test_timesteps_device`

---

#### 1.5 Window Alpha Calculation

**Method**: `get_window_alpha(self, timepoints: torch.FloatTensor)`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 250-267

**Implementation Steps**:

1. **Lookup window boundaries**:
   ```python
   t_win_start, t_win_end = self.time_windows.lookup_window(timepoints)
   t_win_len = t_win_end - t_win_start
   t_interval = timepoints - t_win_start  # Note: negative value
   ```

2. **Get alpha values at boundaries**:
   ```python
   idx_start = (t_win_start * num_train_timesteps - 1).long()
   alphas_cumprod_start = self.alphas_cumprod[idx_start]
   
   idx_end = torch.clamp((t_win_end * num_train_timesteps - 1).long(), min=0)
   alphas_cumprod_end = self.alphas_cumprod[idx_end]
   ```

3. **Compute gamma**:
   ```python
   alpha_cumprod_s_e = alphas_cumprod_start / alphas_cumprod_end
   gamma_s_e = alpha_cumprod_s_e ** 0.5
   ```

4. **Return tuple**:
   ```python
   return t_win_start, t_win_end, t_win_len, t_interval, gamma_s_e, alphas_cumprod_start, alphas_cumprod_end
   ```

**Tests to Pass**: `test_get_window_alpha`

---

#### 1.6 Denoising Step

**Method**: `step(self, model_output, timestep, sample, return_dict=True)`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 269-341

**Implementation Steps**:

1. **Normalize timestep**:
   ```python
   t_c = timestep / self.config.num_train_timesteps
   ```

2. **Get window parameters**:
   ```python
   t_s, t_e, _, c_to_s, gamma_s_e, alphas_cumprod_start, alphas_cumprod_end = self.get_window_alpha(t_c)
   ```

3. **Prediction type handling**:

   **For "ddim_eps"**:
   ```python
   pred_epsilon = model_output
   lambda_s = (alphas_cumprod_end / alphas_cumprod_start)**0.5
   eta_s = (1-alphas_cumprod_end)**0.5 - (alphas_cumprod_end / alphas_cumprod_start * (1-alphas_cumprod_start))**0.5
   
   lambda_t = (lambda_s * (t_e - t_s)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
   eta_t = (eta_s * (t_e - t_c)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
   
   pred_win_end = lambda_t * sample + eta_t * pred_epsilon
   pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))
   ```

   **For "diff_eps"**:
   ```python
   pred_epsilon = model_output
   lambda_s = 1 / gamma_s_e
   eta_s = -1 * (1 - gamma_s_e**2)**0.5 / gamma_s_e
   
   lambda_t = (lambda_s * (t_e - t_s)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
   eta_t = (eta_s * (t_e - t_c)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
   
   pred_win_end = lambda_t * sample + eta_t * pred_epsilon
   pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))
   ```

   **For "velocity"**:
   ```python
   pred_velocity = model_output
   ```

4. **Compute next timestep**:
   ```python
   idx = torch.argwhere(torch.where(self.timesteps == timestep, 1, 0))
   prev_step = self.timesteps[idx+1] if (idx+1) < len(self.timesteps) else 0
   dt = (prev_step - timestep) / self.config.num_train_timesteps
   dt = dt.to(sample.device, sample.dtype)
   ```

5. **Update sample**:
   ```python
   prev_sample = sample + dt * pred_velocity
   ```

6. **Return result**:
   ```python
   if not return_dict:
       return (prev_sample,)
   return PeRFlowSchedulerOutput(prev_sample=prev_sample, pred_original_sample=None)
   ```

**Tests to Pass**: `test_step_shape`, `test_step_return_dict`, `test_full_loop_no_noise`, `test_step_with_different_prediction_types`

---

#### 1.7 Noise Addition

**Method**: `add_noise(self, original_samples, noise, timesteps)`

**Reference**: `Project/PeRFlow/src/scheduler_perflow.py` lines 344-366

**Implementation Steps**:

1. **Prepare alpha values**:
   ```python
   alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
   timesteps = timesteps.to(original_samples.device) - 1  # indexing from 0
   ```

2. **Get sqrt alpha values**:
   ```python
   sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
   sqrt_alpha_prod = sqrt_alpha_prod.flatten()
   while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
       sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
   ```

3. **Get sqrt one minus alpha**:
   ```python
   sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
   sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
   while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
       sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
   ```

4. **Add noise**:
   ```python
   noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
   return noisy_samples
   ```

**Tests to Pass**: `test_add_noise_shape`, `test_full_loop_with_noise`

---

#### 1.8 Helper Methods

**Method**: `scale_model_input(self, sample, timestep=None)`

**Implementation**:
```python
return sample  # No scaling needed for PeRFlow
```

**Method**: `__len__(self)`

**Implementation**:
```python
return self.config.num_train_timesteps
```

**Tests to Pass**: `test_scale_model_input`, `test_scheduler_length`

---

### Phase 2: ODE Solver Implementation

#### 2.1 PFODESolver Initialization

**File**: `diffusers/src/diffusers/schedulers/pfode_solver.py`

**Reference**: `Project/PeRFlow/src/pfode_solver.py` lines 16-25

**Implementation Steps**:

1. **Store parameters**:
   ```python
   self.t_initial = t_initial
   self.t_terminal = t_terminal
   self.scheduler = scheduler
   ```

2. **Calculate step size**:
   ```python
   train_step_terminal = 0
   train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps
   self.stepsize = (t_terminal - t_initial) / (train_step_terminal - train_step_initial)
   ```

**Tests to Pass**: `test_initialization`, `test_initialization_custom_times`

---

#### 2.2 Timestep Generation

**Method**: `get_timesteps(self, t_start, t_end, num_steps)`

**Reference**: `Project/PeRFlow/src/pfode_solver.py` lines 28-40

**Implementation Steps**:

1. **Reshape inputs**:
   ```python
   t_start = t_start[:, None]  # (b,) -> (b, 1)
   t_end = t_end[:, None]
   assert t_start.dim() == 2
   ```

2. **Create timepoints**:
   ```python
   timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
   interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
   timepoints = t_start + interval * timepoints
   ```

3. **Convert to timesteps**:
   ```python
   timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize
   return timesteps.round().long()
   ```

**Tests to Pass**: `test_get_timesteps_shape`, `test_get_timesteps_dtype`, `test_get_timesteps_values_range`

---

#### 2.3 ODE Solving

**Method**: `solve(self, latents, unet, t_start, t_end, prompt_embeds, negative_prompt_embeds, guidance_scale=1.0, num_steps=2, num_windows=1)`

**Reference**: `Project/PeRFlow/src/pfode_solver.py` lines 42-135

**Implementation Steps**:

1. **Setup**:
   ```python
   assert t_start.dim() == 1
   assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
   
   do_classifier_free_guidance = True if guidance_scale > 1 else False
   bsz = latents.shape[0]
   ```

2. **Prepare embeddings**:
   ```python
   if do_classifier_free_guidance:
       prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
   ```

3. **Handle timestep conditioning** (if needed):
   ```python
   timestep_cond = None
   if unet.config.time_cond_proj_dim is not None:
       guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
       timestep_cond = self.get_guidance_scale_embedding(
           guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
       ).to(device=latents.device, dtype=latents.dtype)
   ```

4. **Generate timesteps**:
   ```python
   timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
   timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)
   ```

5. **Denoising loop**:
   ```python
   with torch.no_grad():
       for i in range(num_steps):
           t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
           latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
           latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
           
           # Predict noise
           noise_pred = unet(
               latent_model_input,
               t,
               encoder_hidden_states=prompt_embeds,
               timestep_cond=timestep_cond,
               return_dict=False,
           )[0]
           
           # Apply guidance
           if do_classifier_free_guidance:
               noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
               noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
           
           # Compute previous sample (DDIM-style)
           batch_timesteps = timesteps[:, i].cpu()
           prev_timestep = batch_timesteps - timestep_interval
           
           alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
           alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
           for ib in range(prev_timestep.shape[0]):
               alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
           beta_prod_t = 1 - alpha_prod_t
           
           # Move to device
           alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
           alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
           beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)
           
           # Compute x0 and update
           if self.scheduler.config.prediction_type == "epsilon":
               pred_original_sample = (latents - beta_prod_t[:, None, None, None] ** 0.5 * noise_pred) / alpha_prod_t[:, None, None, None] ** 0.5
               pred_epsilon = noise_pred
           elif self.scheduler.config.prediction_type == "v_prediction":
               pred_original_sample = (alpha_prod_t[:, None, None, None]**0.5) * latents - (beta_prod_t[:, None, None, None]**0.5) * noise_pred
               pred_epsilon = (alpha_prod_t[:, None, None, None]**0.5) * noise_pred + (beta_prod_t[:, None, None, None]**0.5) * latents
           else:
               raise ValueError(f"prediction_type {self.scheduler.config.prediction_type} not supported")
           
           pred_sample_direction = (1 - alpha_prod_t_prev[:, None, None, None]) ** 0.5 * pred_epsilon
           latents = alpha_prod_t_prev[:, None, None, None] ** 0.5 * pred_original_sample + pred_sample_direction
   
   return latents
   ```

**Tests to Pass**: `test_solve_basic`, `test_solve_with_guidance`, `test_solve_different_num_steps`

---

#### 2.4 PFODESolverSDXL Implementation

**Reference**: `Project/PeRFlow/src/pfode_solver.py` lines 144-283

**Key Differences**:

1. **Additional time embeddings**:
   ```python
   def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
       add_time_ids = list(original_size + crops_coords_top_left + target_size)
       add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
       return add_time_ids
   ```

2. **SDXL-specific conditioning**:
   ```python
   add_text_embeds = pooled_prompt_embeds
   add_time_ids = torch.cat([
       self._get_add_time_ids((resolution, resolution), (0, 0), (resolution, resolution), dtype)
       for _ in range(bsz)
   ]).to(device)
   
   if do_classifier_free_guidance:
       prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
       add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
       add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
   ```

3. **Modified UNet call**:
   ```python
   added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
   noise_pred = unet(
       latent_model_input,
       t,
       encoder_hidden_states=prompt_embeds,
       timestep_cond=timestep_cond,
       added_cond_kwargs=added_cond_kwargs,
       return_dict=False,
   )[0]
   ```

**Tests to Pass**: All PFODESolverSDXL tests (8 methods)

---

### Phase 3: Utility Functions Implementation

#### 3.1 Delta Weight Merging

**File**: `diffusers/src/diffusers/schedulers/utils_perflow.py`

**Reference**: `Project/PeRFlow/src/utils_perflow.py` lines 10-18

**Implementation**:

```python
def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()
    assert unet_weights.keys() == delta_weights.keys()
    
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe
```

**Tests to Pass**: `test_merge_delta_weights_*` (6 tests)

---

#### 3.2 Loading Delta Weights

**Reference**: `Project/PeRFlow/src/utils_perflow.py` lines 21-58

**Implementation Steps**:

1. **Check for delta_weights.safetensors**:
   ```python
   if os.path.exists(os.path.join(model_path, "delta_weights.safetensors")):
       print("### delta_weights exists, loading...")
       delta_weights = OrderedDict()
       with safe_open(os.path.join(model_path, "delta_weights.safetensors"), framework="pt", device="cpu") as f:
           for key in f.keys():
               delta_weights[key] = f.get_tensor(key)
   ```

2. **Or check for merged weights**:
   ```python
   elif os.path.exists(os.path.join(model_path, "diffusion_pytorch_model.safetensors")):
       print("### merged_weights exists, loading...")
       # Load merged weights
       # Compute delta = merged - base
       # Save delta weights
   ```

3. **Merge into pipeline**:
   ```python
   pipe = merge_delta_weights_into_unet(pipe, delta_weights)
   return pipe
   ```

**Tests to Pass**: `test_load_delta_weights_*` (4 tests)

---

#### 3.3 DreamBooth Loading

**Reference**: `Project/PeRFlow/src/utils_perflow.py` lines 62-77

**Implementation**:

```python
def load_dreambooth_into_pipeline(pipe, sd_dreambooth):
    assert sd_dreambooth.endswith(".safetensors")
    
    state_dict = {}
    with safe_open(sd_dreambooth, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    unet_config = {}
    for key in pipe.unet.config.keys():
        if key != 'num_class_embeds':
            unet_config[key] = pipe.unet.config[key]
    
    pipe.unet.load_state_dict(convert_ldm_unet_checkpoint(state_dict, unet_config), strict=False)
    pipe.vae.load_state_dict(convert_ldm_vae_checkpoint(state_dict, pipe.vae.config))
    pipe.text_encoder = convert_ldm_clip_checkpoint(state_dict, text_encoder=pipe.text_encoder)
    
    return pipe
```

**Tests to Pass**: `test_load_dreambooth_*` (3 tests)

---

## Implementation Order

### Recommended Sequence

1. **Week 1: Core Scheduler**
   - Implement TimeWindows class
   - Implement betas_for_alpha_bar
   - Implement PeRFlowScheduler.__init__
   - Run basic initialization tests

2. **Week 2: Scheduler Methods**
   - Implement set_timesteps
   - Implement get_window_alpha
   - Implement scale_model_input and __len__
   - Run timestep generation tests

3. **Week 3: Denoising**
   - Implement step method for all prediction types
   - Implement add_noise
   - Run full loop tests

4. **Week 4: ODE Solvers**
   - Implement PFODESolver
   - Implement PFODESolverSDXL
   - Run solver tests

5. **Week 5: Utilities & Integration**
   - Implement utility functions
   - Integration testing
   - Performance optimization

---

## Testing Strategy

### Unit Testing

For each implemented component:

1. Run specific test file:
   ```bash
   python -m pytest tests/schedulers/test_scheduler_perflow.py::PeRFlowSchedulerTest::test_timesteps -v
   ```

2. Run all tests for a class:
   ```bash
   python -m pytest tests/schedulers/test_scheduler_perflow.py -v
   ```

### Integration Testing

After all components are implemented:

1. Test with actual diffusion pipeline
2. Compare outputs with original PeRFlow implementation
3. Validate numerical accuracy
4. Performance benchmarking

### Validation Checklist

- [ ] All 69 test methods pass
- [ ] No NotImplementedError exceptions
- [ ] Numerical outputs match reference implementation
- [ ] Performance meets targets
- [ ] Documentation complete and accurate
- [ ] Code passes linting and type checking

---

## Performance Considerations

### Optimization Opportunities

1. **TimeWindows.lookup_window**: Vectorize batch operations
2. **get_window_alpha**: Cache alpha values for repeated timesteps
3. **PFODESolver.solve**: Use torch.compile for inner loop
4. **Memory**: Use in-place operations where safe

### Benchmarks

Target performance (on V100 GPU):
- SD 1.5: 10 steps in < 2 seconds
- SDXL: 10 steps in < 5 seconds

---

## Debugging Guide

### Common Issues

1. **Shape mismatches**: Check tensor broadcasting in window calculations
2. **Numerical instability**: Add epsilon to divisions, check precision
3. **Device mismatches**: Ensure all tensors on same device
4. **Type errors**: Verify dtype consistency throughout pipeline

### Debug Tools

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check intermediate values
print(f"t_win_start: {t_win_start}, t_win_end: {t_win_end}")
print(f"alphas_cumprod_start: {alphas_cumprod_start}")

# Validate shapes
assert sample.shape == expected_shape, f"Got {sample.shape}, expected {expected_shape}"
```

---

## References

### Primary Sources

1. **Original Implementation**: `Project/PeRFlow/src/`
   - `scheduler_perflow.py`: Scheduler reference
   - `pfode_solver.py`: Solver reference
   - `utils_perflow.py`: Utilities reference

2. **Documentation**: `Project/docs/raw/report0/perflow.md`
   - API specifications
   - Design decisions
   - Integration requirements

3. **Test Specifications**: All test files define expected behavior

### Additional Resources

- PeRFlow paper: [Link to arXiv]
- Diffusers documentation: https://huggingface.co/docs/diffusers
- Rectified Flow theory: [Relevant papers]

---

## Success Metrics

### Code Quality

- [ ] All tests pass (69/69)
- [ ] Code coverage > 95%
- [ ] No linting errors
- [ ] Type hints complete
- [ ] Docstrings comprehensive

### Functionality

- [ ] Matches original PeRFlow outputs (tolerance: 1e-5)
- [ ] Compatible with diffusers pipelines
- [ ] Supports all prediction types
- [ ] Works with SD 1.5, SD 2.1, SDXL

### Performance

- [ ] 5-10x faster than standard DDIM at same quality
- [ ] Memory efficient (no significant overhead)
- [ ] GPU utilization > 80%

---

## Contact & Support

For questions during implementation:
- Refer to test files for expected behavior
- Check original PeRFlow code for reference implementation
- Review diffusers existing schedulers for patterns

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-04  
**Status**: Framework Complete, Ready for Implementation
