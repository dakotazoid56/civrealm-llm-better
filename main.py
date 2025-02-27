# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle
import warnings
import gymnasium
import civrealm
from agents import RandomLLMAgent, MistralAgent
from civrealm.freeciv.utils.freeciv_logging import fc_logger
from civrealm.configs import fc_args
from civrealm.envs.freeciv_wrapper.llm_wrapper import LLMWrapper
from agents.utils import print_step, print_action, print_current
from agents import utils

# FIXME: This is a hack to suppress the warning about the gymnasium spaces. Currently Gymnasium does not support hierarchical actions.
warnings.filterwarnings('ignore',
                        message='.*The obs returned by the .* method.*')


def main():
    """
    Main

    Main entry of the program.
    Starts a single-player Freeciv game against rule-based AI.
    """
    env = gymnasium.make('civrealm/FreecivLLM-v0')
    #env = LLMWrapper(env)
    agent = MistralAgent()
    # agent = BaseLangAgent()

    observations, info = env.reset()

    done = False
    step = 0
    while not done:
        try:
            action = agent.act(observations, info)
            observations, reward, terminated, truncated, info = env.step(
                action)
            done = terminated or truncated

            step += 1
            print_step(f'Step: {step}, Turn: {info["turn"]}, ' +
                       f'Reward: {reward}, Terminated: {terminated}, ' +
                       f'Truncated: {truncated}')
        except Exception as e:
            fc_logger.error(repr(e))
            raise e
    env.close()
    '''
    players, tags, turns, evaluations = env.evaluate_game()
    '''
    env.plot_game_scores()
    game_results = env.get_game_results()
    print('game results:', game_results)


if __name__ == '__main__':
    main()
