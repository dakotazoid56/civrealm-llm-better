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


import random
import os
import json
import time
from datetime import datetime
import shutil


from mistralai import Mistral
from civrealm.agents.base_agent import BaseAgent
from civrealm.configs import fc_args

model = "mistral-large-latest"
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

save_directory = os.path.join(os.getcwd(), "saved_dialogues")

class MistralAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        if fc_args["debug.randomly_generate_seeds"]:
            agentseed = os.getpid()
            self.set_agent_seed(agentseed)
        else:
            if "debug.agentseed" in fc_args:
                self.set_agent_seed(fc_args["debug.agentseed"])

        clear_saved_dialogues_folder()  #Remove previous run data

    def act(self, observation, info):
        if info['turn'] != self.turn:
            self.planned_actor_ids = []
            self.turn = info['turn']

        for ctrl_type, actors_dict in info['llm_info'].items():
            for actor_id in actors_dict.keys():
                if actor_id in self.planned_actor_ids:
                    continue
                available_actions = actors_dict[actor_id]['available_actions']
                if available_actions:
                    # Query LLM To Get action_name
                    action_name = self.llm_choose_action_from_actor_info(actors_dict[actor_id])
                    #action_name = self.llm_choose_random_action(available_actions)
                    #action_name = random.choice(available_actions)
                    self.planned_actor_ids.append(actor_id)
                    return (ctrl_type, actor_id, action_name)

    def query_llm(self, prompt):
        """
        Query the LLM with the given prompt and return the generated text.
        """

        while True:
            try:
                response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},]
                )
                break
            except Exception as e:
                # Check if it's the rate limit error by examining the error attributes
                if hasattr(e, 'status_code') and e.status_code == 429:
                    #print("Rate limit exceeded. Retrying in 1 second...")
                    time.sleep(1)  # Wait for 1 second before retrying
                else:
                    raise  # Re-raise if it's not a rate limit error

        return response.choices[0].message.content.strip()
              
    def llm_choose_random_action(self, available_actions):
        """
        Query the LLM with the list of available_actions. The prompt instructs 
        the LLM to pick one action randomly and return it in a structured JSON 
        response. This function then parses the JSON and returns the chosen action.

        :param available_actions: A list of available actions the model can choose from.
        :param client: The AI client used to call the LLM (e.g., an OpenAI client).
        :param model: The LLM model name/identifier to be used (e.g., "gpt-4" or "mistral").
        :return: The chosen action name (str).
        """
        
        # Create a structured prompt asking the model to choose one action from the list
        prompt = f"""
        You are an AI assistant provided with a list of possible actions:
        {available_actions}

        Your task: pick a single action at random from this list and reply using ONLY the following JSON structure:

        {{
        "action_name": "<SELECTED_ACTION>"
        }}

        No additional commentary. No explanations. Return valid JSON only.
        """

        llm_output = self.query_llm(prompt)
        # Extract text from LLM response

        # Try to parse JSON from the LLM's response
        try:
            parsed = json.loads(llm_output)
            action_name = parsed.get("action_name")
            # Validate that the selected action is in the list
            if action_name not in available_actions:
                raise ValueError(f"LLM chose an invalid action: {action_name}")
            print(f"LLM chose action: {action_name}")
        except (json.JSONDecodeError, ValueError, KeyError):
            # If parsing fails or LLM picks an invalid action, 
            # fall back to random choice from the list
            action_name = random.choice(available_actions)
            print(f"LLM failed to parse JSON, falling back to random choice: {action_name}")

        return action_name

    def llm_choose_action_from_actor_info(self, actor):
        """

        """
        available_actions = actor['available_actions']
        actor_name = actor['name']
        
        # Create a structured prompt asking the model to choose one action from the list
        prompt = f"""
        You are an AI playing a Civilization-style game.
        Your task: Achieve Total World Domination. Expand, explore, and multiply as fast as possible.

        You are the following character:
        {json.dumps(actor, indent=4)}

        You must choose an action from the available actions:

        {json.dumps(available_actions, indent=4)}

        **IMPORTANT**: Your response must be a valid JSON object with the following structure:

        ```json
        {{
            "reasoning": "<EXPLANATION_OF_WHY_THIS_ACTION_WAS_CHOSEN>",
            "action_name": "<SELECTED_ACTION>"
        }}
        ```

        - Do not provide any additional commentary.
        - Do not include markdown formatting like ```json.
        - Return only a valid JSON object.

        """

        llm_output = self.query_llm(prompt)
        
        # Extract text from LLM response
        try:
            parsed = json.loads(llm_output.strip())  # Ensure clean JSON parsing
            action_name = parsed.get("action_name")

            # Validate that the selected action is in the list
            if action_name not in available_actions:
                raise ValueError(f"LLM chose an invalid action: {action_name} for {actor_name}")

            print(f"LLM chose action for {actor_name}: {action_name}")

        except (json.JSONDecodeError, ValueError, KeyError):
            # If parsing fails or LLM picks an invalid action, 
            # fall back to random choice from the list
            action_name = random.choice(available_actions)
            print(f"LLM failed to parse JSON, falling back to random choice for {actor_name}: {action_name}")

        # Create the filename with actor_name, turn number, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{actor_name}_Turn{self.turn}_{timestamp}.txt"
        filepath = os.path.join(save_directory, filename)

        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save prompt and llm_output to file
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Prompt:\n{prompt}\n\nLLM Output:\n{llm_output}")

        return action_name

    

def clear_saved_dialogues_folder():
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)  # Remove the directory and all its contents
    os.makedirs(save_directory)  # Recreate an empty directory
