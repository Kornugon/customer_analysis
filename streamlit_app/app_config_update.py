import json
from typing import Any, Dict


class ConfigUpdater:
    def __init__(
            self,
            model_name: str,
            task: str) -> None:
        """
        Initializes the ConfigUpdater with the given model name.

        :param str model_name: The name of the model.\
            Avaliable "RNN" and "Transformer".
        :param str task: Parameter that determine, whether to predict\
            user churn or next event.

        :raises FileNotFoundError: Raises, if the configuration file for the\
            given model name cannot be found.
        """
        self.model = f'{model_name}Model'

        config_files = {
            "RNN": {
                "churn": "rnn_churn_config.json",
                "events": "rnn_events_config.json"
            },
            "Transformer": {
                "churn": "transformer_churn_config.json",
                "events": "transformer_events_config.json"
            }
        }
        pth = f'streamlit_app/config/nn/{config_files[model_name][task]}'

        try:
            with open(pth, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No configuration file found for the model: {model_name}")

        self.json_file = pth

    def update_section(
            self,
            section: str,
            **params: Dict[str, Any]) -> None:
        """
        Updates the given section of the model configuration
        with the provided parameters.

        :param str section: The section of the model configuration to update.
        :param Dict[str, Any] params: The parameters values to update\
            the section with.
        """
        self.data[self.model][section].update(params)

    def get_param(
            self,
            section: str,
            param: str) -> Any:
        """
        Gets the value of the given parameter from the given section
        of the model configuration.

        :param str section: The section of the model configuration.
        :param str param: The parameter to get the value of.

        :return Any: The value of the parameter.
        """
        return self.data[self.model][section].get(param)

    def save(self) -> None:
        """
        Saves the current state of the model configuration
        to its respective .json file.
        """
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=4)

        # NOTE: below print is temporary:
        print("NOTE: Parameters updated sucessfully.")
