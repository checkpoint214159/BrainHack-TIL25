"""
Pipeline Logging Utilities
Standardized print functions for visibility across all pipeline components.
Organized by component level: Orchestrator, Trainer, Simulator, Rollouts.
"""

from datetime import datetime


def _get_timestamp():
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("%H:%M:%S")


def _format_header(title, width=80):
    """Format a section header with separators."""
    return f"\n{'='*width}\n[{_get_timestamp()}] {title}\n{'='*width}"


def _format_subheader(title, width=80):
    """Format a subsection header."""
    return f"\n{'-'*width}\n[{_get_timestamp()}] {title}\n{'-'*width}"


# ============================================================================
# ORCHESTRATOR LEVEL LOGGING
# ============================================================================

def log_orchestrator_init(orchestrator_hash, root_dir, num_loops, num_policies, db_path):
    """Log orchestrator initialization."""
    print(_format_header("ORCHESTRATOR INITIALIZATION"))
    print(f"  Orchestrator Hash: {orchestrator_hash}")
    print(f"  Root Directory: {root_dir}")
    print(f"  Total Training Loops: {num_loops}")
    print(f"  Number of Unique Policies: {num_policies}")
    print(f"  Database Path: {db_path}")


def log_orchestrator_loop_start(loop_num, total_loops, num_loops_per_policy):
    """Log start of outer training loop."""
    print(_format_subheader(f"ORCHESTRATOR LOOP {loop_num + 1}/{total_loops}"))


def log_policy_training_start(policy_id, num_vec_envs, policy_mapping):
    """Log when training a specific policy."""
    controlled_agents = policy_mapping.count(policy_id)
    print(_format_subheader(f"TRAINING POLICY {policy_id}"))
    print(f"  Number of Vectorized Environments: {num_vec_envs}")
    print(f"  Agents Controlled by Policy {policy_id}: {controlled_agents}")
    print(f"  Simulator Policies Setup: {len([p for p in policy_mapping if p is not None])} controlled, {len([p for p in policy_mapping if p is None])} opponents")


def log_ray_tune_config(num_samples, max_concurrent, perturbation_interval, num_hyperparams):
    """Log Ray Tune configuration."""
    print(_format_subheader("RAY TUNE CONFIGURATION"))
    print(f"  Number of Samples (Parallel Trials): {num_samples}")
    print(f"  Max Concurrent Trials: {max_concurrent}")
    print(f"  Perturbation Interval: every {perturbation_interval} trials")
    print(f"  Total Hyperparameters to Tune: {num_hyperparams}")


def log_orchestrator_results(policy_id, best_result):
    """Log results after orchestrator completes training for a policy."""
    print(_format_subheader(f"POLICY {policy_id} TRAINING COMPLETE"))
    if best_result is not None:
        print(f"  Best Result: {best_result}")
    else:
        print(f"  WARNING: No best result found")


# ============================================================================
# TRAINER LEVEL LOGGING
# ============================================================================

def log_trainer_setup_start(trial_name, trial_code):
    """Log trainer setup beginning."""
    print(_format_header("CUSTOM TRAINER SETUP"))
    print(f"  Trial Name: {trial_name}")
    print(f"  Trial Code: {trial_code}")


def log_trainer_config_merge(num_policies, num_hparams):
    """Log configuration merging."""
    print(_format_subheader("CONFIG MERGING"))
    print(f"  Policies in This Trial: {num_policies}")
    print(f"  Total Hyperparameters: {num_hparams}")


def log_trainer_env_build(train_env_num_envs, eval_env_num_envs, num_policies):
    """Log environment building."""
    print(_format_subheader("ENVIRONMENT BUILDING"))
    print(f"  Train Environment Vectorized Envs: {train_env_num_envs}")
    print(f"  Eval Environment Vectorized Envs: {eval_env_num_envs}")
    print(f"  Policies Using These Environments: {num_policies}")


def log_trainer_simulator_init(num_policies, total_envs, n_steps, db_path):
    """Log simulator initialization."""
    print(_format_subheader("ROLLOUT SIMULATOR INIT"))
    print(f"  Number of Policies: {num_policies}")
    print(f"  Total Environments (vec_envs * agents): {total_envs}")
    print(f"  Steps Per Rollout: {n_steps}")
    print(f"  Database Path: {db_path}")


def log_trainer_callbacks_setup(num_checkpoint_callbacks, num_total_callbacks, eval_freq):
    """Log callback setup."""
    print(_format_subheader("CALLBACKS SETUP"))
    print(f"  Checkpoint Callbacks (per policy): {num_checkpoint_callbacks}")
    print(f"  Total Callbacks: {num_total_callbacks}")
    print(f"  Evaluation Frequency: every {eval_freq} steps")


def log_trainer_step_start(training_iteration, total_timesteps):
    """Log trainer step beginning."""
    print(_format_subheader(f"TRAINER STEP {training_iteration}"))
    print(f"  Total Timesteps to Collect: {total_timesteps}")


def log_trainer_step_results(step_num, all_policy_scores, policy_results):
    """Log trainer step results."""
    print(_format_subheader(f"TRAINER STEP {step_num} RESULTS"))
    print(f"  All Policy Scores (Optimization Metric): {all_policy_scores:.4f}")
    for policy_id, result in policy_results.items():
        print(f"    Policy {policy_id}: {result:.4f}")


# ============================================================================
# SIMULATOR LEVEL LOGGING
# ============================================================================

def log_simulator_init_start(num_policies, vec_envs, n_steps, selfplay):
    """Log simulator initialization starting."""
    print(_format_header("ROLLOUT SIMULATOR INITIALIZATION"))
    print(f"  Number of Policies: {num_policies}")
    print(f"  Vectorized Environments: {vec_envs}")
    print(f"  Steps Per Rollout: {n_steps}")
    print(f"  Self-Play Mode: {selfplay}")


def log_policy_loading(policy_id, source, path_or_db_info):
    """Log policy loading for a specific policy."""
    source_str = f"Loading from {source}: {path_or_db_info}"
    print(f"  Policy {policy_id}: {source_str}")


def log_policy_agent_indexes(policy_id, agent_indexes):
    """Log policy to agent indexing."""
    print(f"  Policy {policy_id} Controls Agent Indices: {agent_indexes}")


def log_policy_initialization_summary(num_initialized, num_from_disk, num_from_db, num_fresh):
    """Log summary of policy initialization."""
    print(_format_subheader("POLICY INITIALIZATION SUMMARY"))
    print(f"  Total Policies Initialized: {num_initialized}")
    print(f"  - From Disk: {num_from_disk}")
    print(f"  - From Database: {num_from_db}")
    print(f"  - Freshly Initialized: {num_fresh}")


def log_learn_start(total_timesteps, num_policies, num_steps, total_envs):
    """Log learn loop beginning."""
    print(_format_header("LEARN LOOP START"))
    print(f"  Target Total Timesteps: {total_timesteps}")
    print(f"  (Note: Includes vec_envs factor of {total_envs})")
    print(f"  Number of Policies: {num_policies}")
    print(f"  Steps Per Rollout: {num_steps}")
    print(f"  Total Envs (vec_envs × agents): {total_envs}")
    print(f"  Calculated Rollout Steps: {num_steps * total_envs}")


def log_learn_iteration(current_timesteps, total_timesteps, num_policies_trained):
    """Log progress during learn loop."""
    progress = (current_timesteps / total_timesteps) * 100 if total_timesteps > 0 else 0
    print(_format_subheader(f"LEARN ITERATION PROGRESS"))
    print(f"  Current Timesteps: {current_timesteps}/{total_timesteps} ({progress:.1f}%)")
    print(f"  Policies Trained This Iteration: {num_policies_trained}")


def log_policy_training(policy_id, num_timesteps, total_timesteps):
    """Log individual policy training."""
    progress = (num_timesteps / total_timesteps) * 100 if total_timesteps > 0 else 0
    print(f"  [Policy {policy_id}] Training... ({progress:.1f}%)")


def log_learn_end(total_timesteps, num_policies, total_time):
    """Log learn loop end."""
    print(_format_header("LEARN LOOP END"))
    print(f"  Total Timesteps Collected: {total_timesteps}")
    print(f"  Policies Trained: {num_policies}")
    print(f"  Total Time: {total_time:.2f}s")


# ============================================================================
# ROLLOUT COLLECTION LOGGING
# ============================================================================

def log_collect_rollouts_start(n_rollout_steps, num_policies):
    """Log rollout collection beginning."""
    print(_format_subheader("COLLECT ROLLOUTS"))
    print(f"  Target Rollout Steps: {n_rollout_steps}")
    print(f"  Policies Collecting Rollouts: {num_policies}")


def log_rollout_step(step_num, n_rollout_steps, num_policies_stepping):
    """Log individual rollout step."""
    progress = (step_num / n_rollout_steps) * 100 if n_rollout_steps > 0 else 0
    print(f"    Step {step_num}/{n_rollout_steps} ({progress:.1f}%) - {num_policies_stepping} policies stepping", end='\r')


def log_evaluation_checkpoint(step_num, eval_freq, timesteps):
    """Log when evaluation happens during rollout."""
    print(f"\n    ⚡ EVALUATION CHECKPOINT (step {step_num}, every {eval_freq} steps, {timesteps} total timesteps)")


def log_collect_rollouts_end(n_steps_collected, total_rewards_per_policy, continue_training):
    """Log rollout collection end."""
    print(_format_subheader("COLLECT ROLLOUTS END"))
    print(f"  Total Steps Collected: {n_steps_collected}")
    for policy_id, rewards in total_rewards_per_policy.items():
        if len(rewards) > 0:
            mean_reward = sum(sum(r) for r in rewards) / len(rewards) if len(rewards) > 0 else 0
            print(f"  Policy {policy_id} Mean Reward: {mean_reward:.4f}")
    print(f"  Continue Training: {continue_training}")


def log_observation_format(step_num, num_policies, total_obs_keys):
    """Log observation formatting."""
    print(f"    Formatted observations for {num_policies} policies ({total_obs_keys} observation keys)")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_error(component, message):
    """Log an error message."""
    print(_format_header(f"ERROR IN {component}"))
    print(f"  {message}")


def log_warning(component, message):
    """Log a warning message."""
    print(_format_subheader(f"WARNING in {component}"))
    print(f"  {message}")


def log_info(message):
    """Log a general info message."""
    print(f"[{_get_timestamp()}] ℹ️  {message}")
