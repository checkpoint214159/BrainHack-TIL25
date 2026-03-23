from collections import defaultdict
import re

def generate_policy_agent_indexes(selfplay, n_envs, policy_mapping):
        """
        A crucial method that produces a dictionary telling us what policies map to which index of
        the collections a policy controls.

        For instance, presume we have 2 vector environments. This function will produce something like
        {
            0: [0, 4],
            1: [1, 5],
            2: [2, 6],
            3: [3, 7],
        }
        to tell which indexes a policy's observations and actions should be mapped to.

        The behaviour of this function will change if we are conducting self-play. 
        When in self play, we would like, for any given game, only one agent operating under a policy
        currently being trained. This is to ensure that performance is more fairly benchmarked against
        static opponents retrieved from the database.
        However, the above is made more complicated, when we have one policy mapping to multiple agents m.
        In this case, the environment should be vectorized m times, and for each agent, play against static
        opponents, but all of these m agents share the same set of currently evolving parameters.

        An example of how it will change under selfplay is this:

        Normal run, policy_mapping [1, 0, 0, 0], agent_roles [0, 1, 2, 3], 3 vec envs.
        Return: {
            0: [1, 2, 3, 5, 6, 7, 9, 10, 11], 
            1: [0, 4, 8],
        }

        Selfplay run, policy_mapping [None, 0, 0, 0], agent_roles [0, 1, 2, 3].  Because during selfplay
        we only learn one policy at a time, and 1 -> None from an earlier component.
        Vector environments will be automatically set to 3 by earlier components too.
        Return: {
            0: [1, 6, 11]  # yes, for 3 vector envs! Once again we are only training one policy at a time, per vector env.
        }
        For the first vec env, actions will be [None, action_from_polid_0, None, None] -> 1st index occupied
        For the second vec env, actions will be [None, None, action_from_polid_0, None] -> 2st index occupied + 4 = 6th global index
        For the first vec env, actions will be [None, None, None action_from_polid_0] -> 3rd index occupied + 8 = 11th global index
        Action sent off to env: [
            None,
            action_from_polid_0,  -> 1st index
            None,
            None,
            None,
            None,
            action_from_polid_0,  -> 6th index
            None,
            None,
            None,
            None,
            action_from_polid_0,  -> 11th index
        ]
        """
        policy_indexes = {}
        n_policy_mappings = policy_mapping * n_envs # just extend this list.
  
        if selfplay:
            assert len(set(policy_mapping)) <= 2, 'Assertion failed. Generate_policy_agent_indexes currently ' \
                'expects up to only two unique elements in policy_mapping: a unique policy number, and None.'
            
            for polid in list(set(policy_mapping)):
                if polid is not None:
                    indices = [idx for idx, val in enumerate(policy_mapping) if val == polid]
                    for env_idx, idx in enumerate(indices):
                        indices[env_idx] = idx + env_idx * len(policy_mapping)
                    policy_indexes[polid] = indices

        else:  # if not selfplay, easy list duplicate
            for idx, polid in enumerate(n_policy_mappings):
                if polid is not None:
                    if polid not in policy_indexes:
                        policy_indexes[polid] = [idx]
                    else:
                        policy_indexes[polid].append(idx)

        return policy_indexes

    
def split_dict_by_prefix(source_dict):
    pattern = re.compile(r"^(\d+)/(.+)$")  # this is chatgpt'd. i have no idea of regex works.
    grouped = defaultdict(dict)

    for key, value in source_dict.items():
        match = pattern.match(key)
        if match:
            prefix, suffix = match.groups()
            grouped[int(prefix)][suffix] = value
    return dict(grouped)  # convert defaultdict to regular dict
    


def replace_and_report(base: dict, override: dict, merge=False) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in base and base[key] != value:
            print(f"Overriding key '{key}': {base[key]} -> {value}")
            merged[key] = value
        elif merge:
            print(f"Adding new key '{key}': {value}")
            merged[key] = value

    return merged