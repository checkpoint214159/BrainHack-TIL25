import numpy as np
import re
import os
from collections import defaultdict
from ray import tune
from copy import deepcopy
# the 3 components we will use
from simulator import RLRolloutSimulator
from selfplay_env import build_env
from custom_callbacks import CustomEvalCallback, CustomCheckpointCallback
from selfplay_env import CustomRewardNames
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement,
)
from utils import replace_and_report, split_dict_by_prefix


# rewards dict given by default by TIL. we evaluate on this
STD_REWARDS_DICT = {
    CustomRewardNames.GUARD_CAPTURES: 50,
    CustomRewardNames.SCOUT_CAPTURED: -50,
    CustomRewardNames.SCOUT_RECON: 1,
    CustomRewardNames.SCOUT_MISSION: 5,
    CustomRewardNames.WALL_COLLISION: 0,
    CustomRewardNames.STATIONARY_PENALTY: 0,
    CustomRewardNames.SCOUT_STEP_EMPTY_TILE: 0,
}


class CustomTrainer(tune.Trainable):
        """
        Our very own policy trainer class.
        Pass in the whole ray_hyp_config during class initialization, and we'll handle the rest.

        Trainable handles creation of train-related semantics, including the environments
        """

        def setup(self, ray_hyp_config: dict, base_config):
            """
            Sets up training, for a new experiment created by tune. 
            Merges base and ray config, replacing default hyperparams and specifications from base
            with whatever ray generates.

            Args:
                - ray_hyp_config: Hyperparam ray_hyp_config passed down to us from the ray gods
                - base_config: base ray_hyp_config from the yaml file (see tune.with_parameters to see
                    how it got passed down here).
            """
            # setup call. each new iteration, take configs given to us
            # and override copies of our defaults that were defined in our init.

            self.root_dir = base_config.train.root_dir
            self.experiment_name = base_config.train.experiment_name

            self._training_config = base_config.train
            self._env_config = base_config.env
            self._train_env_config = self._env_config.train
            self._eval_env_config = self._env_config.eval
            self._policies_config = base_config.policies
            
            # deepcopy to avoid imploding original configs (for whatever reason)
            training_config = deepcopy(self._training_config)
            train_env_config = deepcopy(self._train_env_config)
            eval_env_config = deepcopy(self._eval_env_config)
            policies_config = deepcopy(self._policies_config)

            training_config = replace_and_report(training_config, ray_hyp_config)
            train_env_config = replace_and_report(train_env_config, ray_hyp_config)
            
            # we want to be more careful for the eval_env_config, since we keep it controlled at
            # 100 iters. just hardcode replace the frame stack size
            if eval_env_config.frame_stack_size != ray_hyp_config['frame_stack_size']:
                eval_env_config.frame_stack_size = ray_hyp_config['frame_stack_size']

            # merge policy configurations; override old with new per policy basis.
            policies_hparams = split_dict_by_prefix(ray_hyp_config)
            assert len(policies_hparams) == len(policies_config), 'Assertion failed, mismatching number of policies as specified by tune configuration,' \
                'and number of policies under the policies section. Failing gracefully.'

            for polid, incoming_policy_config in policies_hparams.items():
                policies_config[polid] = replace_and_report(policies_config[polid], incoming_policy_config, merge=True)
            self.policies_config = policies_config

            REWARDS_DICT = {
                CustomRewardNames.GUARD_CAPTURES: ray_hyp_config.get('guard_captures'),
                CustomRewardNames.SCOUT_CAPTURED: ray_hyp_config.get('scout_captured'),
                CustomRewardNames.SCOUT_RECON: ray_hyp_config.get('scout_recon'),
                CustomRewardNames.SCOUT_MISSION: ray_hyp_config.get('scout_mission'),
                CustomRewardNames.WALL_COLLISION: ray_hyp_config.get('wall_collision'),
                CustomRewardNames.STATIONARY_PENALTY: ray_hyp_config.get('stationary_penalty'),
                CustomRewardNames.SCOUT_STEP_EMPTY_TILE: ray_hyp_config.get('scout_step_empty_tile'),
                CustomRewardNames.LOOKING: ray_hyp_config.get('looking'),
            }
            
            # initialize the environment configurations, but throw in some other things from the main
            # configuration part that are required too.
            _, train_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                db_path=base_config.db_path,
                env_config=train_env_config,
            )
            _, eval_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=STD_REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                db_path=base_config.db_path,
                env_config=eval_env_config,
            )
            
            self.agent_roles = list(base_config.agent_roles)
            self.simulator_policies = list(base_config.simulator_policies)
            trial_name = self.trial_name
            trial_code = trial_name[:-6]

            self.eval_log_path = f"{self.root_dir}/ppo_logs/{trial_code}/{trial_name}"
            self.simulator = RLRolloutSimulator(
                selfplay=True,
                db_path=base_config.db_path,
                train_env=train_env,
                train_env_config=train_env_config,
                policies_config=self.policies_config,
                policy_mapping=self.simulator_policies,
                tensorboard_log=self.eval_log_path,
                callbacks=None,
                n_steps=training_config.n_steps,
                use_action_masking=training_config.use_action_masking,
                verbose=0,
            )
            self.total_timesteps = training_config.training_iters * train_env.num_envs
            eval_freq = int(self.total_timesteps / training_config.num_evals / train_env.num_envs)
            eval_freq = int(max(eval_freq, training_config.n_steps))
            training_config.eval_freq = eval_freq

            checkpoint_callbacks = [
                CustomCheckpointCallback(
                    save_freq=eval_freq,
                    polid=polid,
                    db_path=base_config.db_path,
                    save_path=f"{self.root_dir}/checkpoints/{trial_code}/{trial_name}",
                    name_prefix=f"{self.experiment_name}"
                ) for polid in self.policies_config
            ]
            no_improvement = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=training_config.no_improvement,
                min_evals=int(training_config.num_evals) * 0.25,
                verbose=1
            )

            eval_callback = CustomEvalCallback(
                in_bits=True if eval_env_config.binary == 'binary' else False,  # TODO this is really bad code
                log_path=self.eval_log_path,
                agent_roles=self.agent_roles,
                policy_mapping=self.simulator_policies,
                eval_env_config=eval_env_config,
                training_config=training_config,
                eval_env=eval_env,                    
                callback_after_eval=no_improvement,
                deterministic=False,
            )

            self.callbacks = checkpoint_callbacks
            self.eval_callback = eval_callback
            

        def step(self):
            self.simulator.learn(
                total_timesteps=self.total_timesteps,
                callbacks=self.callbacks,
                eval_callback=self.eval_callback,
            )

            logging_dict = defaultdict()
            mean_policy_scores = []

            for polid in self.policies_config:
                path = os.path.join(self.eval_log_path, "evaluations", f"policy_id_{polid}.npz")
                thing = np.load(path)
                mean_scores = np.mean(thing['results'], axis=-1)
                max_mean_eval = np.max(mean_scores)
                max_idx = np.argmax(mean_scores)
                mean_policy_scores.append(max_mean_eval)
                best_timestep = thing['timesteps'][max_idx]
                logging_dict.setdefault(f'policy_{polid}_best_result', max_mean_eval)
                logging_dict.setdefault(f'policy_{polid}_best_timestep', best_timestep)

            all_policy_scores = sum(mean_policy_scores)
            logging_dict.setdefault('all_policy_scores', all_policy_scores)
            self.logging_dict = logging_dict
            self.logging_dict.update({'step': 1})

            return logging_dict
        
        def save_checkpoint(self, tmp_checkpoint_dir):
            path = os.path.join(tmp_checkpoint_dir, "state.npz")
            np.savez(
                path,
                **self.logging_dict
            )
            return tmp_checkpoint_dir

        def load_checkpoint(self, tmp_checkpoint_file):
            # print('tmp_checkpoint_file??', tmp_checkpoint_file)
            state = np.load(tmp_checkpoint_file)
            self.iter = state["step"]
            self.all_policy_scores = state['all_policy_scores']
