import argparse

import os
import shutil
import hashlib
import uuid

from collections import defaultdict
from ray import tune
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy

from ray.tune.schedulers import PopulationBasedTraining
from selfplay_trainer import CustomTrainer

# re-define reward names to pass into our env-builder.


def parse_args():
    parser = argparse.ArgumentParser(description="Example script with config file option")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the config file"
    )
    return parser.parse_args()

config_path = 'standard_config.yaml'

def interpret_search_space(cfg_node):

    typ = cfg_node.get("type")
    if typ == "uniform":
        return tune.uniform(cfg_node.min, cfg_node.max)
    elif typ == "loguniform":
        return tune.loguniform(cfg_node.min, cfg_node.max)
    elif typ == "choice":
        return tune.choice(cfg_node.choices)
    else:
        raise AssertionError('Unknown type provided')


def generate_8char_hash():
    unique_str = str(uuid.uuid4())  # Generate a unique string
    hash_object = hashlib.sha256(unique_str.encode())  # Hash it
    short_hash = hash_object.hexdigest()[:8]  # Take the first 8 characters
    return short_hash

class SelfPlayOrchestrator:
    """
    A wrapper around ray tune for training. This orchestrates agent selection and policy loading,
    continuing to loop until some stopping condition (TODO decide this). 

    While stopping condition has not passed:
    1. Select an agent, load in its corresponding policy.
    2. Initialize ray tune and trainable. Tune allows for us to try how different hyperparameters fare in a trial.
    3. Obtain the best result, checkpoint it into the DB, and loop back to 1.

    Each orchestrator will generate its own unique 8-size hash, and that will be the working directory
    of all checkpoints, ray results, and tensorboard logs.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize orchestration.
        Most important configurations are what agents we are training, and what agents are part of the environment.
        """
        self.config = config
        assert len(self.config.agent_roles) == len(self.config.policy_mapping), 'Assertion failed. agent_roles in the config '\
            'must have the same length as policy_mapping in the config.'
        self.policy_mapping = config.policy_mapping
        self.hash = generate_8char_hash()
        self.config.train.root_dir = os.path.join(self.config.train.root_dir, f'Orchestrator_{self.hash}')
        if not os.path.exists(self.config.db_name):
            db_path = os.path.join(self.config.train.root_dir, self.config.db_name)
        else:
            db_path = self.config.db_name
        # if path to db is not in our root directory, copy the db file into it.
        file_path = Path(db_path).resolve()
        parent_dir = Path(self.config.train.root_dir).resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_path.relative_to(parent_dir)
        except ValueError:
            # do a copy
            db_path = parent_dir / file_path.name
            shutil.copy2(file_path, db_path)

        self.config.db_path = db_path
        self.num_loops = self.config.train.num_loops * len(self.config.agent_roles)
        self.agent_roles = self.config.agent_roles

    def commence(self):
        """
        Commences training.
        For all collections of selectable policies, build the trainable and call ray tune to optimize it.
        """
        # we loop over agent_roles to know how many agents there are. no stopping condition for now.
        for _ in range(self.num_loops):
            for polid in list(set(self.policy_mapping)):
                tmp_config = deepcopy(self.config)  # deepcopy to avoid mutating underlying config, for whatever reason.

                # boolean mask of npcs: 0 represents the selected agent, 1 represents an agent controlled by the environment.
                policies_controlled_here = [polid] # integer indexes of policy controlled here ( for now, just one. )
                num_vec_envs = self.policy_mapping.count(polid)
                tmp_config.env.train.num_vec_envs = num_vec_envs
                tmp_config.env.eval.num_vec_envs = num_vec_envs

                # for the policies not controlled by the orchestrator-simulator, mark None.
                self.simulator_policies = deepcopy(self.policy_mapping)
                for idx, polid in enumerate(self.policy_mapping):
                    if polid not in policies_controlled_here:
                        self.simulator_policies[idx] = None

                tmp_config.simulator_policies = self.simulator_policies

                experiment_name = tmp_config.train.experiment_name

                tune_config = tmp_config.tune

                # first, create the hyperparam search space and population based training.
                # this will optimize the hyperparams within each run.
                # there are some that can be opted to be policy specific, and some that are environment specific.
                # things that control environment run length or steps before policies train are examples of these.
                policy_independent_hparams = {
                    "n_steps": interpret_search_space(tune_config.n_steps),
                    "frame_stack_size": interpret_search_space(tune_config.frame_stack_size),
                    "novice": interpret_search_space(tune_config.novice),
                    "num_iters": interpret_search_space(tune_config.num_iters),
                    "guard_captures": interpret_search_space(tune_config.guard_captures),
                    "scout_captured": interpret_search_space(tune_config.scout_captured),
                    "scout_recon": interpret_search_space(tune_config.scout_recon),
                    "scout_mission": interpret_search_space(tune_config.scout_mission),
                    "scout_step_empty_tile": interpret_search_space(tune_config.scout_step_empty_tile),
                    "stationary_penalty": interpret_search_space(tune_config.stationary_penalty),
                    "looking": interpret_search_space(tune_config.looking),
                    "wall_collision": interpret_search_space(tune_config.wall_collision),
                    "distance_penalty": interpret_search_space(tune_config.distance_penalty),
                }
                # extract only the hparams relevant to the policy(ies) we are training.
                tune_config.policies = {
                    polid: v for polid, v in tune_config.policies.items() if polid in policies_controlled_here
                }
                policy_dependent_hparams = [{
                    f"{polid}/{k}": interpret_search_space(v) for k, v in policy_config.items()
                } for polid, policy_config in tune_config.policies.items()]

                # prune those configurations
                tmp_config.policies = {
                    polid: v for polid, v in tmp_config.policies.items() if polid in policies_controlled_here
                }

                # merge everything
                merged = {}
                [merged.update(d) for d in policy_dependent_hparams]
                merged.update(policy_independent_hparams)

                pbt = PopulationBasedTraining(
                        time_attr="training_iteration",
                        metric="all_policy_scores",
                        mode="max",
                        perturbation_interval=4,  # every n trials
                        hyperparam_mutations=merged)
                
                trainable_cls = tune.with_parameters(CustomTrainer, base_config=tmp_config) # this is where the edited config
                # gets passed to the trainable. environment is initialized within trainable.
            
                tuner = tune.Tuner(
                    tune.with_resources(trainable_cls,
                        resources=dict(tmp_config.resources)
                    ),
                    tune_config=tune.TuneConfig(
                            scheduler=pbt,
                            num_samples=100,
                            max_concurrent_trials=4,
                    ),
                    run_config=tune.RunConfig(
                        name='test',
                        storage_path=f"{tmp_config.train.root_dir}/ray_results/{experiment_name}",
                        verbose=1,
                        stop={"training_iteration": 1},
                    )
                )

                results = tuner.fit()
                print('get best results', results.get_best_result())


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        exit(1)
    else:
        print(f"Using config file: {config_path}")

    base_config = OmegaConf.load(config_path)
    orchestrator = SelfPlayOrchestrator(config=base_config)
    orchestrator.commence()
