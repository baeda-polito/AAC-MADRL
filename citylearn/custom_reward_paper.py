from typing import Any, List, Mapping, Union, Optional
import numpy as np
from citylearn.reward_function import RewardFunction

class CustomReward(RewardFunction):
    """Calculates custom user-defined multi-agent reward.

    Reward is the :py:attr:`net_electricity_consumption_emission`
    for entire district if central agent setup otherwise it is the
    :py:attr:`net_electricity_consumption_emission` each building.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]], beta: float, gamma: float) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        self.beta = beta
        self.gamma = gamma
        pricing = [o['electricity_pricing'] for o in observations] 
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations] 
        district_electricity_consumption = sum(net_electricity_consumption)
        reward_agent = [(1-self.beta)*min((pricing[i]*(-1*net_electricity_consumption[i]))**3, 0) for i in range(len(net_electricity_consumption))]
        reward_list = 10**2*(reward_agent - self.beta*(district_electricity_consumption**2)/len(net_electricity_consumption)**self.gamma)

        if self.central_agent:
            reward = [reward_list.sum()]
        else:
            reward = list(reward_list)

        return reward
