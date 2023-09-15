# Copyright (C) 2023  The Freeciv-gym project
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

import os
import time
import threading

from freeciv_gym.freeciv.utils.language_agent_utility import make_action_list_readable, get_action_from_readable_name

from .language_agent import LanguageAgent
from .workers import AzureGPTWorker


class ParallelAutoGPTAgent(LanguageAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dialogue_dir = os.path.join(
            os.getcwd(), 'agents/civ_autogpt/saved_dialogues/')

    def initialize_workers(self):
        self.workers = {}

    def add_entity(self, entity_type, entity_id):
        self.workers[(entity_type, entity_id)] = AzureGPTWorker()

    def remove_entity(self, entity_type, entity_id):
        del self.workers[(entity_type, entity_id)]

    def process_observations_and_info(self, observations, info):
        self.observations = observations
        self.info = info

    def get_obs_input_prompt(self, ctrl_type, actor_name, actor_dict,
                             available_actions):
        current_unit_obs = actor_dict['observations']['minimap']
        if ctrl_type == "city":
            available_actions += ["'keep activity'"]
        prompt = f'The {ctrl_type} is {actor_name}, observation is {current_unit_obs}. Your available action list is {available_actions}.'
        system_message = self.info['llm_info'].get("message", "")
        system_message = ("Game scenario message is: "
                          if system_message else "") + system_message
        return system_message + prompt

    def make_single_decision(self, ctrl_type, actor_id, actor_dict):
        worker = self.workers[(ctrl_type, actor_id)]
        actor_name = actor_dict['name']
        available_actions = make_action_list_readable(
            actor_dict['available_actions'])
        obs_input_prompt = self.get_obs_input_prompt(ctrl_type, actor_name,
                                                     actor_dict,
                                                     available_actions)
        print(f'Current {ctrl_type}: {actor_name}')
        exec_action_name = worker.choose_action(obs_input_prompt,
                                                available_actions)
        print(f'Action chosen for {actor_name}:', exec_action_name)
        exec_action_name = get_action_from_readable_name(exec_action_name)
        if exec_action_name and exec_action_name != "keep activity":
            self.chosen_actions.put((ctrl_type, actor_id, exec_action_name))

        worker.save_dialogue_to_file(
            os.path.join(
                self.dialogue_dir,
                f"dialogue_T{self.info['turn'] + 1:03d}_{actor_id}_at_{time.strftime('%Y.%m.%d_%H:%M:%S')}.txt"
            ))

    def make_decisions(self):
        threads = []
        for ctrl_type in self.info['llm_info'].keys():
            for actor_id, actor_dict in self.info['llm_info'][ctrl_type].items(
            ):
                thread = threading.Thread(target=self.make_single_decision,
                                          args=(ctrl_type, actor_id,
                                                actor_dict))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()
