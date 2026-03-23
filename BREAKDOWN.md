# RL Self-Play Training Pipeline - Complete Breakdown

## Overview

This is a **Multi-Agent Reinforcement Learning (MARL) self-play training framework** for competitive environments. The pipeline trains multiple policies using Ray Tune's Population-Based Training (PBT) scheduler, with agents competing against static checkpoints from a centralized SQLite database.

**Game Context**: 4-agent Pac-Man style arena where a scout collects points while guards try to catch it.

---

## System Architecture

### 1. Entry Point: `selfplay.py` (SelfPlayOrchestrator)

**Purpose**: Orchestrate the entire training loop across multiple policies.

**Key Components**:

```
SelfPlayOrchestrator.__init__()
├─ Loads YAML config (OmegaConf)
├─ Generates unique 8-character hash (e.g., "0bf0b456")
├─ Creates root directory: {root_dir}/Orchestrator_{hash}
├─ Handles DB file (copies external DB into orchestrator dir if needed)
└─ Initializes metadata (policy_mapping, agent_roles, num_loops)

SelfPlayOrchestrator.commence()
├─ Loops over each policy (outer loop: num_loops * num_policies iterations)
├─ For each policy:
│   ├─ Creates temporary config copy
│   ├─ Determines which agents are being trained (polid)
│   ├─ Sets num_vec_envs = count of agents this policy controls
│   ├─ Marks non-controlled agents as None in simulator_policies
│   ├─ Builds PBT hyperparameter search space
│   ├─ Creates Ray Tune Tuner with:
│   │   ├─ Scheduler: PopulationBasedTraining
│   │   ├─ num_samples: 100 (parallel trials)
│   │   ├─ max_concurrent_trials: 4
│   │   └─ storage_path: {root_dir}/ray_results/{experiment_name}
│   └─ Runs tuner.fit() and collects results
```

**Data Flow**:
```
YAML Config
    ↓
Orchestrator (unique hash generated)
    ↓
Creates: {root_dir}/Orchestrator_{hash}/
    ├── selfplay.db (checkpoint database)
    ├── ray_results/{experiment_name}/
    │   └── [Ray trial results]
    └── checkpoints/{trial_code}/{trial_name}/
        └── [Model checkpoint files]
```

**Key Functions**:

- `parse_args()`: Parses `--config` argument
- `interpret_search_space(cfg_node)`: Converts config search space to Ray tune objects (uniform, loguniform, choice)
- `generate_8char_hash()`: Creates unique run identifier

**Outputs**:
- Each orchestrator working directory is unique
- All results/checkpoints/logs isolated under `Orchestrator_{hash}/`

---

### 2. Training Infrastructure: `selfplay_trainer.py` (CustomTrainer)

**Purpose**: Serves as Ray Tune's `Trainable` class. Handles per-trial setup and training loop.

**Initialization (`setup()` method)**:

```
CustomTrainer.setup(ray_hyp_config, base_config)
├─ Merges ray_hyp_config (from PBT) with base_config
├─ Updates hyperparameters for all policies individually
│   └─ Policy-specific: learning_rate, layers, etc.
│   └─ Shared: n_steps, frame_stack_size, rewards, etc.
├─ Calls build_env() twice:
│   ├─ Training environment (num_vec_envs = count of policy's agents)
│   └─ Eval environment (num_vec_envs = count of policy's agents)
├─ Initializes RLRolloutSimulator with:
│   ├─ train_env
│   ├─ policies_config (per-policy hyperparams)
│   ├─ policy_mapping (which agents each policy controls)
│   ├─ callbacks list
│   └─ selfplay=True
├─ Sets up CustomEvalCallback for evaluation
├─ Sets up CustomCheckpointCallback per policy (for DB checkpointing)
└─ Sets up StopTrainingOnNoModelImprovement callback
```

**Training Loop (`step()` method)**:

```
CustomTrainer.step()
├─ Calls simulator.learn(total_timesteps, callbacks, eval_callback)
│   └─ [See simulator.py section for details]
├─ After learning completes, aggregates results:
│   ├─ Loads evaluation scores from .npz files per policy
│   ├─ Computes best result (max mean score)
│   └─ Computes all_policy_scores (sum of best results)
└─ Returns logging_dict with:
    ├─ policy_{polid}_best_result
    ├─ policy_{polid}_best_timestep
    └─ all_policy_scores (PBT optimization metric)
```

**Checkpoint Management (`save_checkpoint()`, `load_checkpoint()`)**: 
- Saves state to `.npz` (step counter, scores)
- Loads state for PBT population-based training resumption

**Key Integration Points**:
- Built using `tune.with_parameters()` to pass `base_config`
- Results aggregated across all policies trained in one trial
- Evaluation happens every `eval_freq` steps (auto-calculated from num_evals)

---

### 3. Rollout Collection & Policy Training: `simulator.py` (RLRolloutSimulator)

**Purpose**: Manages vectorized environment stepping, multi-policy coordination, and policy training.

**Initialization (`__init__()` method)**:

```
RLRolloutSimulator.__init__()
├─ Stores configuration (policies_config, policy_mapping, db_path)
├─ Generates policy_agent_indexes via generate_policy_agent_indexes()
│   └─ Maps each policy to which observation/action indices it controls
│   └─ In selfplay mode: spreads one policy across multiple environments
├─ Creates DummyVecEnv per policy:
│   ├─ Number of dummy envs = number of agents policy controls
│   └─ Used as env for policy initialization (Stable-Baselines3 requirement)
├─ Initializes policies:
│   ├─ For each policy in policies_config:
│   ├─ Checks if 'path' is specified (load from disk) or query DB
│   ├─ If DB query: get_checkpoint_by_policy()
│   │   ├─ If checkpoints exist: load best by mean score
│   │   └─ If no checkpoints: initialize fresh policy
│   └─ Store policy in self.policies[polid]
└─ Sets up callbacks (custom evaluation, checkpointing, early stopping)
```

**Main Learning Loop (`learn()` method)**:

```
RLRolloutSimulator.learn(total_timesteps, callbacks, eval_callback)

Initialize per-policy:
├─ Reset rollout buffers
├─ Reset episode info tracking
└─ Initialize tensorboard logger per policy

While num_timesteps < total_timesteps:
    1. COLLECT ROLLOUTS (collect_rollouts)
    │   ├─ Environment is already vectorized:
    │   │   └─ obs/rewards/dones have shape (num_envs * num_agents, ...)
    │   ├─ For each policy in self.policies:
    │   │   ├─ Call policy.policy.forward(obs[polid])
    │   │   │   └─ Returns actions, values, log_probs
    │   │   ├─ Get action_masks if policy supports it
    │   │   └─ Append actions to step_actions array
    │   ├─ Step environment: env.step(step_actions)
    │   ├─ Format returns back to per-policy format
    │   ├─ Run evaluation callback on_step()
    │   │   └─ May trigger early stopping
    │   ├─ Add to rollout buffers:
    │   │   ├─ obs, action, reward, value, log_prob, episode_start
    │   │   └─ Handles both Dict and Box observation spaces
    │   └─ Return total_rewards per policy, rollout_timesteps, continue_training
    │
    2. TRAIN POLICIES (policy.train())
    │   ├─ For each policy:
    │   │   ├─ policy.train() computes advantages/returns from buffer
    │   │   ├─ Runs mini-batch gradient updates (PPO)
    │   │   ├─ Logs metrics to tensorboard
    │   │   └─ Clears rollout buffer
    │   └─ Update num_timesteps counter
    │
    3. RECORD METRICS
        ├─ Log mean rewards per policy
        ├─ Log FPS, elapsed time
        └─ Dump to tensorboard
```

**Critical Helper: `collect_rollouts()` method**:

```
collect_rollouts(last_obs, n_rollout_steps, eval_callback, callbacks)

Input: last_obs (dict: {polid -> observations per policy})

1. Reformat observations:
   ├─ Dict[str, ndarray] from env → Dict[polid, Dict[str, Tensor]]
   └─ Each policy gets only its relevant observations
   
2. Loop n_rollout_steps times:
   a. Forward pass for each policy:
      ├─ Get actions, values, log_probs from policy.forward()
      ├─ Handle action masking if policy supports it
      └─ Store in all_actions[polid], all_values[polid], etc.
      
   b. Aggregate actions:
      ├─ Map each policy's actions to its agent indices
      ├─ Create full step_actions array (num_envs * num_agents,)
      └─ Step environment
      
   c. Reformat environment returns:
      ├─ obs, rewards, dones, infos back to per-policy dicts
      └─ Convert to tensors for next iteration
      
   d. Callbacks:
      ├─ Update local variables
      ├─ Call eval_callback.on_step()
      │   └─ Returns (continue_training, policy_episode_rewards)
      └─ Call checkpoint callback
      
   e. Add to rollout buffers:
      ├─ Use DummyVecEnv + VecMonitor wrapper
      ├─ Handle discrete vs continuous actions
      └─ Store obs (deepcopy for safety)

3. Return:
   ├─ total_rewards: {polid: [episode_rewards_per_agent]}
   ├─ rollout_timesteps: n_rollout_steps * total_envs
   ├─ continue_training: bool (False if early stopping)
   └─ last_obs: formatted obs for next iteration
```

**Key Data Structure: policy_agent_indexes**

Example with `policy_mapping=[0, 1, 1, 1]`, `n_envs=3`, selfplay=True:
```
policy_agent_indexes = {
    0: [1, 6, 11],        # Policy 0 controls agent 1 (scout) in each of 3 vector envs
    1: [0, 5, 10]         # Policy 1 controls agent 0 (agent 0 of vec env 0) in env 0 only
}
```
This ensures policy 0 is trained across multiple contexts (as scout in different games), while policy 1 faces policy 0 as a static opponent.

---

### 4. Environment: `selfplay_env.py`

**Purpose**: Manages the game environment with self-play support.

**Environment Building (`build_env()` function)**:

```
build_env(reward_names, rewards_dict, policy_mapping, agent_roles,
          self_play, db_path, env_config, env_wrappers=None)

1. Extract config:
   ├─ num_vec_envs: How many parallel environments
   ├─ binary: Binary observations vs full state
   ├─ frame_stack_size: Number of frames to stack
   ├─ top_opponents: Number of best opponents to sample
   └─ other env flags

2. Build original environment:
   ├─ Call modified_env() (PettingZoo environment)
   ├─ Set reward_names, rewards_dict, binary mode, debug mode

3. Apply wrappers in order:
   ├─ AssertOutOfBoundsWrapper (error checking)
   ├─ OrderEnforcingWrapper (protocol checking)
   ├─ aec_to_parallel() (convert to parallel API)
   ├─ frame_stack_v3() (stack N frames, dimension 0)
   ├─ SelfPlayWrapper (if self_play=True) ← KEY COMPONENT
   ├─ pettingzoo_env_to_vec_env_v1() (convert to gym-like VecEnv)
   └─ concat_vec_envs_v1() (parallelize across CPUs)

4. Return: (orig_env, env)
   ├─ orig_env: Single environment (for debugging)
   └─ env: Vectorized environment (for training)
```

**SelfPlayWrapper: Core Self-Play Logic**

```
SelfPlayWrapper (BaseParallelWrapper)

__init__(env, policy_mapping, agent_roles, db_path, top_opponents)
├─ Load database: RL_DB(db_file=db_path, num_roles=len(agent_roles))
├─ Set up agent-to-policy mapping
├─ Load best checkpoints for each role:
│   └─ For each agent role: db.get_checkpoint_by_role(policy, role)
│   └─ Load top_opponents best checkpoint files
├─ Store in self.loaded_policies[agent] (policies to use during play)
└─ Store in self.loaded_desc[agent] (metadata for logging)

reset()
├─ Call env.reset()
└─ Optionally swap opponent policies (every episode or batch)

step(actions)
├─ Receives actions only for trained agent(s)
├─ For non-trained agents (policy_mapping[i] is None):
│   ├─ Load static policy from self.loaded_policies
│   ├─ Get current observations for that agent
│   ├─ Call policy.predict() to get actions
│   └─ Aggregate all actions (trained + static)
├─ Step environment with full action array
└─ Return obs, rewards, dones, infos
```

**Reward System**:

```
CustomRewardNames (StrEnum):
├─ GUARD_WINS: +50 (guards caught scout)
├─ SCOUT_CAPTURED: -50 (scout was caught)
├─ SCOUT_RECON: +1 (scout moves forward towards goal)
├─ SCOUT_MISSION: +5 (scout collects a point)
├─ WALL_COLLISION: 0
├─ AGENT_COLLIDER: +/- (bumped into agent)
├─ STATIONARY_PENALTY: -small (discourage staying still)
└─ ... (more are defined in config)

Rewards are configurable per training trial via YAML
```

**Observation Space**:
- **Binary mode**: 36-dimensional binary vector (view cone with objects encoded as bits)
- **Normal mode**: Full state representation
- **Frame stacking**: Last N observations concatenated (channel-first: `(N*channels, H, W)`)

---

### 5. Callbacks: `custom_callbacks.py`

**Two Main Callback Classes**:

#### A. CustomEvalCallback (Evaluation & Metrics)

```
CustomEvalCallback extends EventCallback

Purpose: Run evaluation episodes and track performance

on_step() [Called every rollout step]:
├─ Check if current step matches eval_freq
├─ If yes:
│   ├─ Run custom_marl_evaluate_policy()
│   ├─ Collect evaluation episodes
│   ├─ Log results to file: {log_path}/evaluations/policy_id_{polid}.npz
│   ├─ Contains:
│   │   ├─ timesteps: [steps when eval was run]
│   │   ├─ results: [mean reward per eval]
│   │   └─ lengths: [episode lengths]
│   └─ Call callback_after_eval (StopTrainingOnNoModelImprovement)
└─ Return (continue_training, policy_episode_rewards)

custom_marl_evaluate_policy():
├─ Reset evaluation environment
├─ For n_eval_episodes:
│   ├─ For each policy:
│   │   ├─ Call policy.predict(obs)
│   │   ├─ Aggregate actions by agent
│   │   └─ Step env
│   ├─ Track episode rewards per policy per role
│   └─ Accumulate results
└─ Average results and save to .npz

Outputs:
├─ Evaluation .npz files saved periodically
├─ Used by CustomTrainer to compute trial scores
└─ Determines best checkpoint per policy
```

#### B. CustomCheckpointCallback (Database Management)

```
CustomCheckpointCallback extends CheckpointCallback

Purpose: Save checkpoints to database with scores

on_step():
├─ Check if current step matches save_freq
├─ If yes:
│   ├─ Get evaluation results from eval_log_path
│   ├─ Compute mean score for this checkpoint
│   ├─ Save model: policy.save(checkpoint_path)
│   ├─ Extract hyperparameters to JSON
│   ├─ Connect to RL_DB
│   ├─ Insert into DB:
│   │   ├─ filepath: path to saved model
│   │   ├─ hyperparameters: JSON string
│   │   ├─ score_{role}: performance per role
│   │   ├─ policy_id, role_id
│   │   └─ timestamp
│   └─ Close DB connection
└─ Continue training
```

**Outputs from Callbacks**:
- **Evaluation scores**: `{log_path}/evaluations/policy_id_{polid}.npz`
- **Checkpoints saved to disk**: `{save_path}/rl_model_{step_count}_steps.zip`
- **Database entries**: SQLite DB with checkpoint metadata and scores

---

### 6. Database Management: `RL_DB` (from `rl/db/db.py`)

**Purpose**: Centralized checkpoint storage and retrieval

**Key Methods**:
```
RL_DB.__init__(db_file, num_roles)
├─ Initialize SQLite connection

set_up_db():
├─ Create tables if not exist
├─ Setup indices

get_checkpoint_by_role(policy, role, shuffle=False):
├─ Query DB for checkpoints of specific policy for specific role
├─ Optionally shuffle (for random sampling)
├─ Return top checkpoints sorted by score

get_checkpoint_by_policy(policy, shuffle=False):
├─ Query DB for all checkpoints of a policy (all roles)

insert_checkpoint(filepath, hyperparameters, score_dict, policy_id, role_id):
├─ Insert new checkpoint with scores

shut_down_db():
├─ Close connection
```

**Database Schema** (inferred):
```
Checkpoints table:
├─ id: unique identifier
├─ filepath: path to .zip model file
├─ hyperparameters: JSON string of config
├─ policy_id: which policy this checkpoint belongs to
├─ role_id: which role (0=scout, 1-3=guards)
├─ score_0, score_1, score_2, score_3: per-role performance scores
├─ timestamp: when checkpoint was saved
└─ (other metadata)
```

---

### 7. Utility Functions: `utils.py`

**Main Function: `generate_policy_agent_indexes()`**

```
generate_policy_agent_indexes(selfplay, n_envs, policy_mapping)

Purpose: Create mapping from policies to observation/action indices

Example 1 (Normal training, policy_mapping=[0, 1, 1, 1], n_envs=3):
Return:
{
    0: [0, 4, 8],           # Policy 0 controls agent 0 in each of 3 vector envs
    1: [1, 2, 3, 5, 6, 7, 9, 10, 11]  # Policy 1 controls agents 1,2,3 in all envs
}

Example 2 (Self-play, policy_mapping=[None, 0, None, None], n_envs=3):
  During training, only policy 0 is being updated
  Environment is vectorized 3 times to train across multiple roles
Return:
{
    0: [1, 6, 11]           # One role per vector env (position 1, then +5 offset, then +5 again)
}

How it's used:
├─ Slice observations: obs_for_pol_0 = all_obs[indexes_0]
├─ Map actions back: all_actions[indexes_0] = pol_0_actions
└─ Track rewards per policy: rewards[indexes_0]
```

**Other Functions**:
- `replace_and_report()`: Merge and report config overrides
- `split_dict_by_prefix()`: Extract policy-specific hyperparams (e.g., "0/lr" -> "lr")

---

## Complete Data Flow Example

```
Starting: python selfplay.py --config selfplay_config.yaml

1. ORCHESTRATOR SETUP
   ├─ Read YAML → OmegaConf object
   ├─ Create SelfPlayOrchestrator(config)
   ├─ Generate hash: "a3a30" → working dir: ./results/Orchestrator_a3a30/
   ├─ Initialize DB at: ./results/Orchestrator_a3a30/selfplay.db
   └─ Start: orchestrator.commence()

2. POLICY LOOP (outer loop)
   For polid=0 (Scout policy):
   ├─ num_vec_envs = 1 (only scout)
   ├─ simulator_policies = [0, None, None, None]
   ├─ Create Ray Tune configuration
   ├─ Launch Tuner with 100 parallel trials, 4 concurrent
   └─ For each trial:
      ├─ Ray spawn trial → CustomTrainer instance

3. TRAINING TRIAL FLOW
   CustomTrainer.setup(ray_hyp_config, base_config):
   ├─ Merge hyperparams
   ├─ Build train_env (1 vec_env for scout)
   ├─ Build eval_env (1 vec_env for scout)
   ├─ Initialize RLRolloutSimulator
   └─ Setup callbacks

   CustomTrainer.step():
   ├─ simulator.learn(total_timesteps)
   
   RLRolloutSimulator.learn():
   ├─ Set up logging, buffers per policy
   ├─ Enter main training loop:
   │  While timesteps < total:
   │  a) collect_rollouts():
   │     ├─ Sample N steps from environment
   │     ├─ For policy 0 (scout):
   │     │  ├─ Get policy 0's observation (index 0)
   │     │  ├─ Call policy.forward() → scout_action
   │     │  └─ Set step_actions[0] = scout_action
   │     ├─ For agents 1,2,3 (guards):
   │     │  ├─ Load static guard policy from DB
   │     │  ├─ Get observations (indices 1,2,3)
   │     │  ├─ Call policy.predict() → guard_actions
   │     │  └─ Set step_actions[1:4] = guard_actions
   │     ├─ env.step(step_actions) → obs, rewards, dones, infos
   │     ├─ Add to policy 0's rollout buffer
   │     └─ Evaluation:
   │        ├─ Run eval episodes every eval_freq steps
   │        ├─ CustomEvalCallback.on_step()
   │        ├─ Save results to: eval_log_path/evaluations/policy_id_0.npz
   │        └─ Returns scores
   │  b) policy.train():
   │     ├─ Compute advantages from rollout buffer
   │     ├─ Do PPO updates (mini-batch gradient descent)
   │     ├─ Clear buffer
   │     └─ Log to tensorboard
   │  c) Checkpoint every save_freq steps:
   │     ├─ CustomCheckpointCallback.on_step()
   │     ├─ Save model to disk
   │     ├─ Get evaluation score from .npz
   │     ├─ Insert into DB with metadata
   │     └─ DB now has checkpoint with score_0 (scout performance)

   After learn() completes:
   ├─ CustomTrainer.step() continues
   ├─ Aggregate eval results from all policies (.npz files)
   ├─ Compute: all_policy_scores = sum of best results
   └─ Return to Ray Tune

   Ray Tune with PBT:
   ├─ Collect results from all 100 trials
   ├─ Use PopulationBasedTraining scheduler
   ├─ Update metric: all_policy_scores
   ├─ Perturbate top trials' hyperparams
   ├─ Save population checkpoint
   └─ Next trial inherits better hyperparams

   For next trial:
   ├─ Ray modifies hyperparams (e.g., learning_rate *= 1.1)
   ├─ Ray calls CustomTrainer.setup() with new ray_hyp_config
   └─ Loop repeats with new hyperparams

4. AFTER POLICY 0 COMPLETES
   ├─ Ray Tune returns best result
   ├─ Best checkpoint loaded from DB (highest score_0)
   └─ Orchestrator loops to policy 1 (guards)

5. POLICY 1 TRAINING (Guards)
   ├─ num_vec_envs = 3 (guards can be in multiple positions)
   ├─ simulator_policies = [None, 1, 1, 1]
   ├─ Load best scout policy from DB (static opponent)
   ├─ Train policy 1 (guards) against fixed scout
   ├─ Same trial flow as policy 0
   ├─ Checkpoints saved with score_1, score_2, score_3 per role
   └─ DB now has guard checkpoints

6. SUBSEQUENT LOOPS
   ├─ Loop 2: Retrain policy 0 again
   │   ├─ Load best policy 1 checkpoints as opponents
   │   └─ Train policy 0 (scout) against better guards
   ├─ Loop 3: Retrain policy 1
   │   ├─ Load best policy 0 checkpoints as opponents
   │   └─ Train policy 1 (guards) against better scout
   └─ Continue for num_loops cycles (coevolution!)

7. OUTPUT STRUCTURE
   ./results/Orchestrator_a3a30/
   ├─ selfplay.db
   │   └─ Contains all checkpoints, scores, hyperparams
   ├─ ray_results/my_experiment/
   │   ├─ trial_0/
   │   │   ├─ params.json
   │   │   ├─ result.json
   │   │   └─ checkpoint_000000/
   │   └─ trial_1/
   ├─ checkpoints/trial_0/trial_0_test_120949-00/
   │   ├─ rl_model_512_steps.zip (policy 0 checkpoint)
   │   └─ rl_model_1024_steps.zip
   └─ ppo_logs/trial_0/trial_0_test_120949-00/
       ├─ evaluations/
       │   ├─ policy_id_0.npz (timesteps, results, lengths)
       │   └─ policy_id_1.npz
       └─ events.out.tfevents (tensorboard logs)
```

---

## Key Output Locations & Contents

### 1. **Database**: `{orchestrator_dir}/selfplay.db`

SQLite database tracking all trained checkpoints.

**Query checkpoints for policy 0, role 0**:
```python
db = RL_DB(db_file='./results/Orchestrator_a3a30/selfplay.db', num_roles=4)
db.set_up_db()
checkpoints = db.get_checkpoint_by_role(policy=0, role=0)  # Scout checkpoints
for ckpt in checkpoints:
    print(ckpt['filepath'])    # Path to .zip file
    print(ckpt['score_0'])     # Scout performance score
    print(ckpt['hyperparameters'])  # JSON of config
```

### 2. **Model Checkpoints**: `{orchestrator_dir}/checkpoints/{trial_code}/{trial_name}/`

Stable-Baselines3 model files saved as `.zip`:
```
rl_model_512_steps.zip
  ├─ Contains policy network weights
  ├─ Can be loaded: policy = PPO.load("path.zip")
  └─ Loaded via DB references in SelfPlayWrapper
```

### 3. **Evaluation Results**: `{orchestrator_dir}/ppo_logs/{trial_code}/{trial_name}/evaluations/`

NumPy files with evaluation metrics:
```
policy_id_0.npz:
  ├─ timesteps: [512, 1024, 1536, ...]  # When evaluations were run
  ├─ results: [[ep1_reward, ep2_reward, ...], [...], ...]
  │           One array per evaluation checkpoint
  │           Each element is mean reward across episodes
  └─ lengths: [[ep1_length, ep2_length, ...], [...], ...]

Usage:
  evals = np.load('policy_id_0.npz')
  mean_rewards = np.mean(evals['results'], axis=-1)  # Mean per eval
  best_idx = np.argmax(mean_rewards)
  best_timestep = evals['timesteps'][best_idx]
  best_score = mean_rewards[best_idx]
```

### 4. **TensorBoard Logs**: `{orchestrator_dir}/ppo_logs/{trial_code}/{trial_name}/`

Event files for visualizing training:
```bash
tensorboard --logdir ./results/Orchestrator_a3a30/ppo_logs
```

Metrics logged per policy:
- `rollout/mean_policy_reward_polid_X`: Mean episode reward
- `time/fps`: Steps per second
- `time/total_timesteps`: Cumulative steps
- Custom metrics from policy config

### 5. **Ray Tune Results**: `{orchestrator_dir}/ray_results/{experiment_name}/`

Ray Tune trial metadata:
```
trial_0/
  ├─ params.json: Final hyperparameters for this trial
  ├─ result.json: Training results per iteration
  └─ checkpoint_000000/: Ray population checkpoint
     └─ state.npz: Trial state for resuming

result.json example:
{
  "training_iteration": 1,
  "all_policy_scores": 145.3,  # This is what PBT optimizes
  "policy_0_best_result": 142.5,
  "policy_0_best_timestep": 1024,
  "date": "2026-03-23 10:30:00"
}
```

---

## Training Dynamics (Self-Play Loop)

**Iteration Pattern** (with `num_loops=3`, 2 policies):

```
Loop 1:
  Policy 0 (Scout)
    ├─ Opponents: Random or previous best guard policies
    ├─ Train, save checkpoints → DB scores[score_1, score_2, score_3]
    └─ Best: policy_0_v1.zip with score_0 = 145

  Policy 1 (Guards)
    ├─ Opponents: Best scout from Loop 1 (policy_0_v1.zip)
    ├─ Train, save checkpoints → DB scores[score_1, score_2, score_3]
    └─ Best: policy_1_v1.zip with score_1 = 95

Loop 2:
  Policy 0 (Scout)
    ├─ Opponents: Best guards from Loop 1 (policy_1_v1.zip) - HARDER!
    ├─ Train (new strategies to overcome better guards)
    └─ Best: policy_0_v2.zip with score_0 = 138 (slightly worse vs new guards)
    
  Policy 1 (Guards)
    ├─ Opponents: Best scout from Loop 2 (policy_0_v2.zip) - HARDER!
    ├─ Train (new strategies)
    └─ Best: policy_1_v2.zip with score_1 = 98

Loop 3:
  Policy 0 (Scout)
    ├─ Opponents: Best guards (policy_1_v2.zip)
    └─ Continue coevolution...

Result: Arms-race behavior where policies push each other to improve
```

---

## Common Queries & Debugging

### Check if training is running:
```bash
# Terminal 1
cd ./results/Orchestrator_{hash}/ppo_logs/{trial_code}/{trial_name}
tensorboard --logdir .

# Open browser to localhost:6006
# Watch: rollout/mean_policy_reward_polid_X increasing over time
```

### Inspect best checkpoint per policy:
```python
from rl.db.db import RL_DB
db = RL_DB('./results/Orchestrator_a3a30/selfplay.db', num_roles=4)
db.set_up_db()

# Get best scout
best_scout = db.get_checkpoint_by_role(policy=0, role=0)[0]
print(f"Scout checkpoint: {best_scout['filepath']}")
print(f"Scout score: {best_scout['score_0']}")

# Get best guard role 1
best_guard = db.get_checkpoint_by_role(policy=1, role=1)[0]
print(f"Guard checkpoint: {best_guard['filepath']}")
print(f"Guard score: {best_guard['score_1']}")

db.shut_down_db()
```

### Manual policy loading & testing:
```python
from otherppos import ModifiedMaskedPPO
policy = ModifiedMaskedPPO.load('./results/.../rl_model_1024_steps.zip')
obs = env.reset()
action, _ = policy.predict(obs)
```

### Check Ray Tune results:
```python
import json
with open('./results/Orchestrator_a3a30/ray_results/my_experiment/trial_0/result.json') as f:
    for line in f:
        result = json.loads(line)
        print(f"Iter {result['training_iteration']}: Score={result['all_policy_scores']}")
```

---

## Summary Table

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Orchestrator** | Coordinate multi-policy training | YAML config | Directory structure + training loop |
| **CustomTrainer** | Per-trial setup & aggregation | ray_hyp_config + base_config | Training results + all_policy_scores |
| **RLRolloutSimulator** | Collect rollouts & train policies | vectorized env + policies | Policy weights updated, rewards logged |
| **collect_rollouts** | Sample environment & agents | env observations | Trajectories in rollout buffers |
| **SelfPlayWrapper** | Load static opponents | db_path + policy_mapping | Step function that includes opponent actions |
| **CustomEvalCallback** | Evaluate policies | eval_env + policies | Evaluation .npz files |
| **CustomCheckpointCallback** | Save & DB tracking | policy + eval_scores | DB entries + checkpoint files |
| **RL_DB** | Checkpoint persistence | SQLite db_file | Query/insert checkpoint metadata |
