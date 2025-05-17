### Tensor returned by `select_action`

- **Shape:** `(batch_size, action_dim)`  
  - `batch_size` – number of parallel environments queried in the same call (1 in the typical single-env case).  
  - `action_dim` – number of robot joints/actuators (`policy.config.action_feature.shape[0]`).  
  - `GeminiPolicy` special case: returns a 1-D tensor of shape `(action_dim,)`.

- **Dtype / Device:** `torch.float32`, located on the same device (CPU/GPU) as the policy.

- **Value range:** already un-normalised and ready for the environment.  
  - Lies within each joint's empirical min/max taken from the dataset.  
  - Some policies clip to `[-1, 1]` before the un-normalisation step; the final values are still within joint limits.

- **Temporal meaning:** contains only the next action step.  
  - Policies that internally compute action chunks (ACT, Diffusion, TD-MPC, etc.) buffer the sequence internally and expose just the current action. 