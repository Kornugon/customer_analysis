import os
import sys
import time
import subprocess
from typing import Any, Optional, Union

import mlflow
import torch.nn as nn
from mlflow.tracking.fluent import ActiveRun
from mlflow.utils.environment import _mlflow_conda_env


class MLFlowManager:
    """
    Class implementing instance of MLFlowManager.
    """

    def __init__(
            self,
            model_name: str,
            config: dict[str, Any]) -> None:
        """
        :param str model_name: Model name.
        :param dict[str, Any] config: Configuration parameters.
        """
        self.model_name = model_name
        self.config = config

        self.id_run = -1
        self.tags = {'phase': "init"}
        self.registered_model_version = -1
        self.registered_model_name = self.model_name

    def start_mlflow_server(
            self) -> Union[subprocess.Popen, None]:
        """
        Starts a local MLflow server as a subprocess with the
        specified port and returns the process object.

        :return Union[subprocess.Popen, None]: The object representing\
            the MLflow server.
        """
        if self.config['use_local_server']:
            local_port = self.config.get('local_server_port', 5000)
            command = ["mlflow", "ui", "--port", str(local_port)]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid)
            time.sleep(2)  # wait for server start up

            for line in iter(process.stdout.readline, b''):
                line = line.decode('utf-8').rstrip()
                if 'Listening at' in line:
                    url = line.split()[-2]
                    print(f'Local MLFlow server listening at: {url}')
                    break

            mlflow.set_tracking_uri(f"http://localhost:{local_port}")
            result = process
        else:
            result = None

        return result

    def start_run(
            self,
            phase: str,
            model_name: str,
            description: str) -> ActiveRun:
        """
        Start a new run.

        :param str phase: The phase of the run.
        :param str model_name: The name of the model.
        :param str description: The description of the run.

        :return ActiveRun: The run object.
        """
        self.tags = {'phase': phase, **self.config['tags']}
        run = mlflow.start_run(
            tags=self.tags, run_name=model_name, description=description)
        self.artifact_uri = run.info.artifact_uri
        self.id_run = run.info.run_id
        self.experiment_id = run.info.experiment_id
        self.conda_env = self._get_conda_env()

        return run

    def log_run(
            self,
            params: dict[str, Any],
            scores: dict[str, float],
            model: nn.Module) -> None:
        """
        Log the parameters, metrics, and model during a run.

        :param dict[str, Any] params: The parameters of the run.
        :param dict[str, float] scores: The metrics of the run.
        :param nn.Module model: The PyTorch model.
        """
        time.sleep(0.5)  # in purpose to not mix print messages
        phase = self.tags["phase"]

        mlflow.log_params(params)
        mlflow.log_metrics(scores)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=self.config['artifact_path'],
            conda_env=self.conda_env
        )
        if self.config['log_model_file']:
            module_path = self._record_lib_file(model)
            mlflow.log_artifact(local_path=module_path)

        if phase == 'predict' and self.config['regis_model_on_predict']:
            model_uri = f'runs:/{self.id_run}/{self.config["artifact_path"]}'
            self.registered_model_name = f'{self.model_name}_{phase}'
            registered_model = mlflow.register_model(
                name=self.registered_model_name,
                model_uri=model_uri,
                tags=self.tags,
                await_registration_for=self.config['await_registration_for'])
            self.registered_model_version = registered_model.version

            if self.config['serve_regis_model_on_predict']:
                self._include_wheel_for_serving(
                    model_version=registered_model.version,
                    model_name=self.registered_model_name,
                    id_run=self.id_run,
                    model_descrip=self.config['regis_model_descrip'])

    def rearch_best_run(
            self,
            model_name: str,
            run_ids: list[str],
            metric: str) -> str:
        """
        Get best run ID, based on given metric name score.

        :param str model_name: Registered model name.
        :param list[str] run_ids: Current runs IDs.
        :param str metric: Metric for filtering logs.

        :return str: Best model ID string.
        """
        runs = mlflow.search_runs()
        matching_runs = runs[
            (runs["tags.mlflow.runName"] == model_name) &
            (runs["run_id"].isin(run_ids))]
        best_run_id = matching_runs.sort_values(
            f"metrics.{metric}", ascending=False)

        return best_run_id.iloc[0]["run_id"]

    def model_staging(
            self,
            model_version: int,
            model_name: str,
            arch_exis_ver: bool = False,
            model_stage: Optional[str] = None,
            model_descrip: Optional[str] = None) -> None:
        """
        Function to transition a registered model to a specified stage
        and update its description.

        :param int model_version: Version of the registered model\
            to transition.
        :param str model_name: Name of the registered model.
        :param bool arch_exis_ver: If True, archive existing versions of the\
            model in the stage being transitioned to. Default: False.
        :param Optional[str] model_stage: Stage to transition the model\
            version to. Options are: ['None', 'Staging', 'Production',\
            'Archived']. Default: None.
        :param Optional[str] model_descrip: Description to update the model\
            version with. Default: None.
        """
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage='None' if not model_stage else model_stage,
            archive_existing_versions=arch_exis_ver)
        if model_descrip:
            client.update_model_version(
                name=model_name,
                version=model_version,
                description=model_descrip)

    def load_model_by_id(
            self,
            model_name: str,
            phase: str,
            model_run_id: str) -> Any:
        """
        Function to load registered model, for given model name and it's ID.

        :param str model_name: Registered model name.
        :param str phase: MLFlow start_run() tags phase to be logged, also\
            registered_model_name part.\
            Here unused, present for consistiency.
        :param str model_run_id: Logged model ID, for filtering.

        :raises ModelNotFound: Raises if a model is not found by\
            provided details.

        :return Any: Logged model. Model Class type.
        """
        runs = mlflow.search_runs()
        matching_runs = runs[
            (runs["tags.mlflow.runName"] == model_name) &
            (runs["run_id"] == model_run_id)]
        if not matching_runs.empty or model_run_id == '-1':
            id = self.best_run_id if model_run_id == '-1' \
                else matching_runs.iloc[0].run_id
            model = mlflow.pytorch\
                .load_model(f'runs:/{id}/model')
        else:
            raise ModelNotFound(f'Not existing ID, in {model_name} runs.')

        return model

    def load_model_by_version(
            self,
            model_name: str,
            phase: str,
            model_version: str) -> Any:
        """
        Function to load registered model for given model name and version.

        :param str model_name: Registered model name.
        :param str phase: MLFlow start_run() tags phase to be logged also\
            registered_model_name part.
        :param str model_version: Registered model version, for filtering.

        :raises ModelNotFound: Raises if a model is not found by\
            provided details.

        :return Any: Registered model. Model Class type.
        """
        try:
            model = mlflow.pytorch\
                .load_model(f"models:/{model_name}_{phase}/{model_version}")
        except Exception:
            raise ModelNotFound(
                f"""Model with name {model_name}_{phase}
                 and stage {model_version} not found.""")

        return model

    def load_model_by_stage(
            self,
            model_name: str,
            phase: str,
            model_stage: str) -> Any:
        """
        Function to load registered model for given model name and stage.

        :param str model_name: Registered model name.
        :param str phase: MLFlow start_run() tags phase to be logged also\
            registered_model_name part.
        :param str model_stage: Registered model stage, for filtering.

        :raises ModelNotFound: Raises if a model is not found by\
            provided details.

        :return Any: Registered model. Model Class type.
        """
        try:
            model = mlflow.pytorch\
                .load_model(f"models:/{model_name}_{phase}/{model_stage}")
        except Exception:
            raise ModelNotFound(
                f"""Model with name {model_name}_{phase}
                 and stage {model_stage} not found.""")

        return model

    def load_model(
            self,
            model_name: str,
            phase: str,
            registered_model_params: dict[str, str]) -> Any:
        """
        Function to load registered model for given model name and
        either run_id, version, or stage.

        :param str model_name: Registered model name.
        :param str phase: MLFlow start_run() tags phase to be logged also\
            registered_model_name part.
        :param dict[str, str] registered_model_params: Dictionary containing\
            optional parameters for filtering. Can contain keys:\
            ['model_run_id', 'model_version', 'model_stage'].

        :raises ModelFilteringParams: Raises with bad parameters configuration.

        :return Any: Registered model. Model Class type.
        """
        load_ways = {
            "model_run_id": self.load_model_by_id,
            "model_version": self.load_model_by_version,
            "model_stage": self.load_model_by_stage
        }
        load_way = {keys: values for keys, values
                    in registered_model_params.items()
                    if keys in load_ways and values is not None}

        if len(load_way) > 1:
            raise ModelFilteringParams(
                """Only one of [model_run_id, model_version, model_stage]
                  can be provided.""")
        elif len(load_way) == 0:
            raise ModelFilteringParams(
                """At least one of [model_run_id, model_version, model_stage]
                  must be provided.""")

        key = next(iter(load_way))

        return load_ways[key](model_name, phase, load_way[key])

    def _get_whl_path(
            self) -> list[str]:
        """
        Get the wheel file path.

        :raises FileNotFoundError: Raises if the library wheel file\
            has not been found.

        :return list[str]: List containing the wheel file path.
        """
        first_choice_path = self.config.get('whl_other_path', "")
        local_path = os.path.abspath(__file__)
        local_path = os.path.abspath(
            os.path.join(local_path, *([os.pardir] * 4))
        ) + self.config['whl_local_path']

        whl_path = next(
            (path for path in [first_choice_path, local_path]
             if os.path.exists(path)), None)

        if whl_path is None:
            raise FileNotFoundError(
                "Library wheel file not found.")

        return [whl_path]

    def _get_conda_env(
            self) -> dict[str, Any]:
        """
        Get the conda environment, including library wheel.

        :return dict[str, Any]: Conda environment.
        """
        conda_env = _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=self._get_whl_path(),
            additional_conda_channels=None)

        return conda_env

    def _record_lib_file(
            self,
            imported_module: nn.Module) -> str:
        """
        Get path to an imported module (model) file location.

        :param nn.Module imported_module: The PyTorch model.

        :return str: A path to model source file.
        """
        module = sys.modules[imported_module.__class__.__module__]

        return os.path.abspath(module.__file__)

    def _include_wheel_for_serving(
            self,
            model_version: str,
            model_name: str,
            id_run: str,
            model_descrip: Optional[str] = None) -> None:
        """
        Helper function - in purpose of preparing registered model for
        model serving in databricks, by including all required dependencies.
        Set enviromet variable to "" (empty string) in purose of reset
        MLFlow default "--only-binary=:all:" value,
        for pip wheel download options. Add library dependencies
        (based on library wheel file) to registered model.
        Creates new version of previously registered model,
        with additional (in fact required) dependencies.

        :param str model_version: Registered model version.
        :param str model_name: Registered model name.
        :param str id_run: Current run ID.
        :param Optional[str] model_descrip: Description to update the model\
            version with. Default: None.
        """
        time.sleep(0.5)  # in purpose to not mix print messages
        _info = 'Adding model libraries to registered model. '
        _info += 'New model version will be created...'
        print(_info)

        mlflow.environment_variables\
            .MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS = \
            mlflow.environment_variables._EnvironmentVariable(
                'MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS', str, "")

        mlflow.models.add_libraries_to_model(
            f'models:/{model_name}/{model_version}',
            id_run)

        default_desc = 'Model prepared for serving. Includes liblary wheel. '
        desc = default_desc + model_descrip if model_descrip else default_desc
        self.registered_model_version = int(model_version) + 1
        client = mlflow.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=self.registered_model_version,
            description=desc)
        self.model_staging(
            model_version=self.registered_model_version,
            model_name=model_name)


class ModelNotFound(Exception):
    """
    Model not found based on given filtering information.
    """


class ModelFilteringParams(Exception):
    """
    No given search params or more than one given.
    """
