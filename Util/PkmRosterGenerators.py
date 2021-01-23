import random
from abc import ABC, abstractmethod
from typing import List

from Util.PkmTeamGenerators import LIST_OF_TYPES, DELTA_HIT_POINTS, DELTA_MOVE_POWER
from Framework.DataConstants import MIN_HIT_POINTS, MOVE_POWER_MIN, DEFAULT_ROSTER_SIZE, DEFAULT_N_MOVES_PKM
from Framework.DataObjects import PkmMoveRoster, PkmRoster, PkmMove, PkmTemplate
from Framework.DataTypes import PkmType
from Framework.StandardPkmMoves import STANDARD_MOVE_ROSTER


class PkmRosterGenerator(ABC):

    @abstractmethod
    def gen_roster(self) -> PkmRoster:
        pass


class RandomPkmRosterGenerator(PkmRosterGenerator):

    def __init__(self, n_moves_pkm: int = DEFAULT_N_MOVES_PKM, pool_size: int = DEFAULT_ROSTER_SIZE):
        self.move_pool: PkmMoveRoster = set(STANDARD_MOVE_ROSTER)
        self.n_moves_pkm = n_moves_pkm
        self.pool_size = pool_size

    def gen_roster(self) -> PkmRoster:
        """
        Generate a random pokemon roster that follows the generator specifications.

        :return: a random pokemon roster.
        """
        roster: List[PkmTemplate] = []
        for i in range(self.pool_size):
            base_move_roster = self.move_pool.copy()
            p_type: PkmType = random.choice(LIST_OF_TYPES)
            max_hp: float = round(random.random() * DELTA_HIT_POINTS + MIN_HIT_POINTS)
            moves = random.sample(list(filter(lambda _m: _m.type == p_type, base_move_roster)), 2)
            for m in moves:
                base_move_roster.remove(m)
            move_roster: List[PkmMove] = moves
            for _ in range(self.n_moves_pkm - 1):
                if random.random() < .2:
                    m_type: PkmType = random.choice(LIST_OF_TYPES)
                    m_power: float = round(random.random() * DELTA_MOVE_POWER + MOVE_POWER_MIN)
                    move = PkmMove(power=m_power, move_type=m_type)
                else:
                    move = random.choice(list(base_move_roster))
                    base_move_roster.remove(move)
                move_roster.append(move)
            random.shuffle(move_roster)
            roster.append(PkmTemplate(set(move_roster), p_type, max_hp))
        return set(roster)
