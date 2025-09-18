from typing import Any, List, Mapping, Union, Optional
import numpy as np
from citylearn.reward_function import RewardFunction

class ComfortConsumptionDistrictRewardFixed(RewardFunction):
    """
    Reward = (1 - beta) * (comfort_i + consumption_i) + beta * district
    - comfort_i: 0 dentro banda; fuori banda = -|Δ|^2
    - consumption_i: min(((-1)*net_i)^3, 0)  # penalizza solo import
    - district: -(sum_i net_i)^2              # penalità quadratica sul distretto
    """

    def __init__(self, env_metadata: Mapping[str, Any], beta: float = 0.5, band: Optional[float] = None):
        super().__init__(env_metadata)
        self.beta = float(beta)
        self.band = band  # se None, usa o['comfort_band']

    # ---------- comfort helper ----------
    def _comfort_term(self, o: Mapping[str, Union[int, float]]) -> float:
        hvac_mode = int(o.get('hvac_mode', 0))
        Tin = float(o['indoor_dry_bulb_temperature'])
        band = self.band if self.band is not None else float(o['comfort_band'])
        if hvac_mode in (1, 2):
            sp = float(
                o['indoor_dry_bulb_temperature_cooling_set_point'] if hvac_mode == 1
                else o['indoor_dry_bulb_temperature_heating_set_point']
            )
            lo, hi = sp - band, sp + band
            if lo <= Tin <= hi:
                return 0.0
            delta = Tin - sp
            return -(delta ** 2)
        else:
            sp_c = float(o['indoor_dry_bulb_temperature_cooling_set_point'])
            sp_h = float(o['indoor_dry_bulb_temperature_heating_set_point'])
            assert sp_c == sp_h, "Set point di riscaldamento e raffreddamento devono essere uguali in modalità off"
            sp = sp_c
            lo, hi = sp - band, sp + band
            if lo <= Tin <= hi:
                return 0.0
            delta = Tin - sp
            return -(delta ** 2)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        nets = [float(o['net_electricity_consumption']) for o in observations]

        comforts = [self._comfort_term(o) for o in observations]
        consumptions = [min(((-1.0) * net) ** 3, 0.0) for net in nets]
        
        district_net = sum(nets)
        district = -(district_net ** 2)
        n_agents = len(observations)
        district = district / (n_agents **3.5)
        rewards = [(1.0 - self.beta) * (comforts[i] + consumptions[i]) + self.beta * district
                   for i in range(n_agents)]

        if self.central_agent:
            return [float(sum(rewards))]
        else:
            return rewards
