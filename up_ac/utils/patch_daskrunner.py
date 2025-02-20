from __future__ import annotations
from smac.runner.abstract_runner import AbstractRunner
from smac.facade.abstract_facade import AbstractFacade
from pathlib import Path
import dask
from dask.distributed import Client, Future, wait

from typing import Callable
import time

import joblib
from typing_extensions import Literal

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from ConfigSpace import Configuration

from smac.callback.callback import Callback
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
import smac.intensifier.abstract_intensifier as aiai  # import AbstractIntensifier
from smac.main.config_selector import ConfigSelector
from smac.model.abstract_model import AbstractModel
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.runner.target_function_script_runner import TargetFunctionScriptRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger, setup_logging

from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.model.random_forest.random_forest import RandomForest
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.runhistory.encoder.encoder import RunHistoryEncoder

from types import MethodType
import logging

logger = get_logger(__name__)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


def print_config_changes_new(
    incumbent: Configuration | None,
    challenger: Configuration | None,
    logger: logging.Logger,
) -> None:
    """Compares two configurations and prints the differences."""
    if incumbent is None or challenger is None:
        return

    params = sorted(
        [
            (param, incumbent[param], challenger[param])
            if param in incumbent.keys()
            else (param, None, challenger[param])
            for param in challenger.keys()
        ]
    )
    for param in params:
        if param[1] != param[2]:
            logger.info("--- %s: %r -> %r" % param)
        else:
            logger.debug("--- %s Remains unchanged: %r", param[0], param[1])


aiai.print_config_changes = print_config_changes_new


class DaskParallelRunner_new(AbstractRunner):

    """Interface to submit and collect a job in a distributed fashion. DaskParallelRunner is
    intended to comply with the bridge design pattern. Nevertheless, to reduce the amount of code
    within single-vs-parallel implementations, DaskParallelRunner wraps a BaseRunner object which
    is then executed in parallel on `n_workers`.

    This class then is constructed by passing an AbstractRunner that implements
    a `run` method, and is capable of doing so in a serial fashion. Next,
    this wrapper class uses dask to initialize `N` number of AbstractRunner that actively wait of a
    TrialInfo to produce a RunInfo object.

    To be more precise, the work model is then:

    1. The intensifier dictates "what" to run (a configuration/instance/seed) via a TrialInfo object.
    2. An abstract runner takes this TrialInfo object and launches the task via
       `submit_run`. In the case of DaskParallelRunner, `n_workers` receive a pickle-object of
       `DaskParallelRunner.single_worker`, each with a `run` method coming from
       `DaskParallelRunner.single_worker.run()`
    3. TrialInfo objects are run in a distributed fashion, and their results are available locally to each worker. The
       result is collected by `iter_results` and then passed to SMBO.
    4. Exceptions are also locally available to each worker and need to be collected.

    Dask works with `Future` object which are managed via the DaskParallelRunner.client.

    Parameters
    ----------
    single_worker : AbstractRunner
        A runner to run in a distributed fashion. Will be distributed using `n_workers`.
    patience: int, default to 5
        How much to wait for workers (seconds) to be available if one fails.
    dask_client: Client | None, defaults to None
        User-created dask client, which can be used to start a dask cluster and then attach SMAC to it. This will not
        be closed automatically and will have to be closed manually if provided explicitly. If none is provided
        (default), a local one will be created for you and closed upon completion.
    """

    def __init__(
        self,
        single_worker: AbstractRunner,
        patience: int = 5,
        dask_client: Client | None = None,
    ):
        super().__init__(
            scenario=single_worker._scenario,
            required_arguments=single_worker._required_arguments,
        )

        # The single worker to hold on to and call run on
        self._single_worker = single_worker

        # The list of futures that dask will use to indicate in progress runs
        self._pending_trials: list[Future] = []

        # Dask related variables
        self._scheduler_file: Path | None = None
        self._patience = patience

        self._client: Client
        self._close_client_at_del: bool

        if dask_client is None:
            dask.config.set({"distributed.worker.daemon": False})
            self._close_client_at_del = True
            self._client = Client(
                n_workers=self._scenario.n_workers,
                processes=True,
                threads_per_worker=1,
                local_directory=str(self._scenario.output_directory),
                timeout=300
            )

            if self._scenario.output_directory is not None:
                self._scheduler_file = Path(self._scenario.output_directory, ".dask_scheduler_file")
                self._client.write_scheduler_file(scheduler_file=str(self._scheduler_file))
        else:
            # We just use their set up
            self._client = dask_client
            self._close_client_at_del = False

    def submit_trial(self, trial_info: TrialInfo, **dask_data_to_scatter: dict[str, Any]) -> None:
        """This function submits a configuration embedded in a ``trial_info`` object, and uses one of
        the workers to produce a result locally to each worker.

        The execution of a configuration follows this procedure:

        #. The SMBO/intensifier generates a `TrialInfo`.
        #. SMBO calls `submit_trial` so that a worker launches the `trial_info`.
        #. `submit_trial` internally calls ``self.run()``. It does so via a call to `run_wrapper` which contains common
           code that any `run` method will otherwise have to implement.

        All results will be only available locally to each worker, so the main node needs to collect them.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.

        dask_data_to_scatter: dict[str, Any]
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.
        """
        # Check for resources or block till one is available
        if self.count_available_workers() <= 0:
            logger.debug("No worker available. Waiting for one to be available...")
            wait(self._pending_trials, return_when="FIRST_COMPLETED")
            self._process_pending_trials()

        # Check again to make sure that there are resources
        if self.count_available_workers() <= 0:
            logger.warning("No workers are available. This could mean workers crashed. Waiting for new workers...")
            time.sleep(self._patience)
            if self.count_available_workers() <= 0:
                raise RuntimeError(
                    "Tried to execute a job, but no worker was ever available."
                    "This likely means that a worker crashed or no workers were properly configured."
                )

        # At this point we can submit the job
        trial = self._client.submit(self._single_worker.run_wrapper, trial_info=trial_info, **dask_data_to_scatter)
        self._pending_trials.append(trial)

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:  # noqa: D102
        self._process_pending_trials()
        while self._results_queue:
            yield self._results_queue.pop(0)

    def wait(self) -> None:  # noqa: D102
        if self.is_running():
            wait(self._pending_trials, return_when="FIRST_COMPLETED")

    def is_running(self) -> bool:  # noqa: D102
        return len(self._pending_trials) > 0

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
        **dask_data_to_scatter: dict[str, Any],
    ) -> tuple[StatusType, float | list[float], float, dict]:  # noqa: D102
        return self._single_worker.run(
            config=config, instance=instance, seed=seed, budget=budget, **dask_data_to_scatter
        )

    def count_available_workers(self) -> int:
        """Total number of workers available. This number is dynamic as more resources
        can be allocated.
        """
        return sum(self._client.nthreads().values()) - len(self._pending_trials)

    def close(self, force: bool = False) -> None:
        """Closes the client."""
        if self._close_client_at_del or force:
            self._client.close()

    def _process_pending_trials(self) -> None:
        """The completed trials are moved from ``self._pending_trials`` to ``self._results_queue``.
        We make sure pending trials never exceed the capacity of the scheduler.
        """
        # In code check to make sure we don't exceed resource allocation
        if self.count_available_workers() < 0:
            logger.warning(
                "More running jobs than resources available. "
                "Should not have more pending trials in remote workers "
                "than the number of workers. This could mean a worker "
                "crashed and was not able to be recovered by dask. "
            )

        # Move the done run from the worker to the results queue
        done = [trial for trial in self._pending_trials if trial.done()]
        for trial in done:
            self._results_queue.append(trial.result())
            self._pending_trials.remove(trial)

    def __del__(self) -> None:
        """Makes sure that when this object gets deleted, the client is terminated. This
        is only done if the client was created by the dask runner.
        """
        if self._close_client_at_del:
            self.close()


class AbstractFacade_new(AbstractFacade):

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable | str | AbstractRunner,
        *,
        model: AbstractModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_maximizer: AbstractAcquisitionMaximizer | None = None,
        initial_design: AbstractInitialDesign | None = None,
        random_design: AbstractRandomDesign | None = None,
        intensifier: aiai.AbstractIntensifier | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
        config_selector: ConfigSelector | None = None,
        logging_level: int | Path | Literal[False] | None = None,
        callbacks: list[Callback] = [],
        overwrite: bool = False,
        dask_client: Client | None = None,
    ):
        setup_logging(logging_level)

        if model is None:
            model = self.get_model(scenario)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(scenario)

        if acquisition_maximizer is None:
            acquisition_maximizer = self.get_acquisition_maximizer(scenario)

        if initial_design is None:
            initial_design = self.get_initial_design(scenario)

        if random_design is None:
            random_design = self.get_random_design(scenario)

        if intensifier is None:
            intensifier = self.get_intensifier(scenario)

        if multi_objective_algorithm is None and scenario.count_objectives() > 1:
            multi_objective_algorithm = self.get_multi_objective_algorithm(scenario=scenario)

        if runhistory_encoder is None:
            runhistory_encoder = self.get_runhistory_encoder(scenario)

        if config_selector is None:
            config_selector = self.get_config_selector(scenario)

        # Initialize empty stats and runhistory object
        runhistory = RunHistory(multi_objective_algorithm=multi_objective_algorithm)

        # Set the seed for configuration space
        scenario.configspace.seed(scenario.seed)

        # Set variables globally
        self._scenario = scenario
        self._model = model
        self._acquisition_function = acquisition_function
        self._acquisition_maximizer = acquisition_maximizer
        self._initial_design = initial_design
        self._random_design = random_design
        self._intensifier = intensifier
        self._multi_objective_algorithm = multi_objective_algorithm
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._config_selector = config_selector
        self._callbacks = callbacks
        self._overwrite = overwrite

        # Prepare the algorithm executer
        runner: AbstractRunner
        if isinstance(target_function, AbstractRunner):
            runner = target_function
        elif isinstance(target_function, str):
            runner = TargetFunctionScriptRunner(
                scenario=scenario,
                target_function=target_function,
                required_arguments=self._get_signature_arguments(),
            )
        else:
            runner = TargetFunctionRunner(
                scenario=scenario,
                target_function=target_function,
                required_arguments=self._get_signature_arguments(),
            )

        # In case of multiple jobs, we need to wrap the runner again using DaskParallelRunner
        if (n_workers := scenario.n_workers) > 1 or dask_client is not None:
            if dask_client is not None:
                logger.warning(
                    "Provided `dask_client`. Ignore `scenario.n_workers`, directly set `n_workers` in `dask_client`."
                )
            else:
                available_workers = joblib.cpu_count()
                if n_workers > available_workers:
                    logger.info(f"Workers are reduced to {available_workers}.")
                    n_workers = available_workers

            # We use a dask runner for parallelization
            runner = DaskParallelRunner_new(single_worker=runner, dask_client=dask_client)

        # Set the runner to access it globally
        self._runner = runner

        # Adding dependencies of the components
        self._update_dependencies()

        # We have to update our meta data (basically arguments of the components)
        self._scenario._set_meta(self.meta)

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

        # Finally we configure our optimizer
        self._optimizer = self._get_optimizer()
        assert self._optimizer

        # Register callbacks here
        for callback in callbacks:
            self._optimizer.register_callback(callback)

        # Additionally, we register the runhistory callback from the intensifier to efficiently update our incumbent
        # every time new information are available
        self._optimizer.register_callback(self._intensifier.get_callback(), index=0)


class AlgorithmConfigurationFacade_new(AbstractFacade_new):
    @staticmethod
    def get_model(  # type: ignore
        scenario: Scenario,
        *,
        n_trees: int = 10,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 20,
        bootstrapping: bool = True,
        pca_components: int = 4,
    ) -> RandomForest:
        """Returns a random forest as surrogate model.

        Parameters
        ----------
        n_trees : int, defaults to 10
            The number of trees in the random forest.
        ratio_features : float, defaults to 5.0 / 6.0
            The ratio of features that are considered for splitting.
        min_samples_split : int, defaults to 3
            The minimum number of data points to perform a split.
        min_samples_leaf : int, defaults to 3
            The minimum number of data points in a leaf.
        max_depth : int, defaults to 20
            The maximum depth of a single tree.
        bootstrapping : bool, defaults to True
            Enables bootstrapping.
        pca_components : float, defaults to 4
            Number of components to keep when using PCA to reduce dimensionality of instance features.
        """
        return RandomForest(
            configspace=scenario.configspace,
            n_trees=n_trees,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            bootstrapping=bootstrapping,
            log_y=False,
            instance_features=scenario.instance_features,
            pca_components=pca_components,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> EI:
        """Returns an Expected Improvement acquisition function.

        Parameters
        ----------
        scenario : Scenario
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        return EI(xi=xi)

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
    ) -> LocalAndSortedRandomSearch:
        """Returns local and sorted random search as acquisition maximizer."""
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        max_config_calls: int = 2000,
        max_incumbents: int = 10,
    ) -> Intensifier:
        """Returns ``Intensifier`` as intensifier. Supports budgets.

        Parameters
        ----------
        max_config_calls : int, defaults to 3
            Maximum number of configuration evaluations. Basically, how many instance-seed keys should be evaluated at
            maximum for a configuration.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """
        return Intensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=max_incumbents,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        additional_configs: list[Configuration] = [],
    ) -> DefaultInitialDesign:
        """Returns an initial design, which returns the default configuration.

        Parameters
        ----------
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        return DefaultInitialDesign(
            scenario=scenario,
            additional_configs=additional_configs,
        )

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        probability: float = 0.5,
    ) -> ProbabilityRandomDesign:
        """Returns ``ProbabilityRandomDesign`` for interleaving configurations.

        Parameters
        ----------
        probability : float, defaults to 0.5
            Probability that a configuration will be drawn at random.
        """
        return ProbabilityRandomDesign(probability=probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        objective_weights: list[float] | None = None,
    ) -> MeanAggregationStrategy:
        """Returns the mean aggregation strategy for the multi objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for averaging the objectives in a weighted manner. Must be of the same length as the number of
            objectives.
        """
        return MeanAggregationStrategy(
            scenario=scenario,
            objective_weights=objective_weights,
        )

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        """Returns the default runhistory encoder."""
        return RunHistoryEncoder(scenario)


def patch_daskrunner(facade):

    return AlgorithmConfigurationFacade_new
