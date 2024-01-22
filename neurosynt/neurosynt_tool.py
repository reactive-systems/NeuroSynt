# import multiprocessing
import logging
import os
import platform
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import psutil
from ml2.ltl import DecompLTLSpec
from ml2.ltl.ltl_spec.ltl_spec_dataset import LTLSpecDataset
from ml2.pipelines.loggers.csv_dataset_logger import CSVToDatasetLogger
from ml2.pipelines.samples.portfolio_sample import PortfolioSample
from ml2.tools.grpc_service import GRPCService
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynProblem
from tqdm import tqdm


class NeuroSynt:
    def __init__(self) -> None:
        self.sem = threading.Semaphore()

    def system_info(self) -> Dict[str, str]:
        try:
            info = {}
            info["platform"] = platform.system()
            info["platform-release"] = platform.release()
            info["platform-version"] = platform.version()
            info["architecture"] = platform.machine()
            info["processor"] = platform.processor()
            info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
            info["cpu-physical-cores"] = psutil.cpu_count(logical=False)
            info["cpu-total-cores"] = psutil.cpu_count(logical=True)
            return info
        except Exception as e:
            logging.exception(e)
            return {}

    def stringify_class(self, config: Dict) -> Dict:
        config = deepcopy(config)
        for tool_kind in config.keys():
            config[tool_kind]["class"] = str(config[tool_kind]["class"])
        return config

    def thread(
        self,
        sample: PortfolioSample,
        problem: ToolLTLSynProblem,
        tool: GRPCService,
        tool_kind: str,
    ):
        start_time = time.time()
        assert getattr(tool, "synthesize", None) is not None
        result_obj = tool.synthesize(problem)  # type: ignore
        end_time = time.time()
        self.sem.acquire()
        sample.add_result(result=result_obj, name=tool_kind, time=end_time - start_time)
        self.sem.release()

    def benchmark(
        self,
        tools: Dict[str, Tuple[GRPCService, Dict[str, Any]]],
        config: Dict,
        dataset: str = "sc-1-f",
        output: str = "benchmark",
        auto_version: bool = False,
        upload: bool = False,
        add_to_wandb: bool = False,
        sample: Optional[int] = None,
    ):
        benchmark: LTLSpecDataset = LTLSpecDataset.load("ltl-spec/" + dataset, name_as_id=True)  # type: ignore
        if sample is not None:
            benchmark.sample(sample)

        # preprocess auto version in artifact
        logger = CSVToDatasetLogger(
            name=output,
            project="ltl-syn",
            auto_version=auto_version,
        )

        logger.metadata = {
            "config": self.stringify_class(config),
            "system_info": self.system_info(),
            "dataset": "ltl-spec/" + dataset,
        }
        slurm_environ = dict((k, v) for k, v in os.environ.items() if k.startswith("SLURM"))
        if len(slurm_environ) != 0:
            logger.metadata["slurm"] = slurm_environ

        # TODO add metric
        for spec in tqdm(benchmark.generator(), total=benchmark.size):
            result = self.synthesize(spec, all_results=True, tools=tools)
            logger.add(sample=result)

        logger.save(add_to_wandb=add_to_wandb, upload=upload)

    def synthesize(
        self,
        spec: DecompLTLSpec,
        tools: Dict[str, Tuple[GRPCService, Dict[str, Any]]],
        all_results: bool = False,
    ) -> PortfolioSample:
        sample = PortfolioSample(inp=spec)
        threads = {}
        for tool_kind, val in tools.items():
            tool, tool_run_args = val
            syn_problem = ToolLTLSynProblem(parameters=tool_run_args, specification=spec)
            thread = threading.Thread(
                target=self.thread, args=(sample, syn_problem, tool, tool_kind)
            )
            thread.start()
            threads[tool_kind] = thread

        if all_results:
            for _, thread in threads.items():
                thread.join()
        else:
            while True:
                if not any([thread.is_alive() for _, thread in threads.items()]):
                    break
                if not all([thread.is_alive() for _, thread in threads.items()]):
                    self.sem.acquire()
                    has_solution = False
                    for r in sample.results:
                        if hasattr(r.result, "status"):
                            if r.result.status.to_int() == 0 or r.result.status.to_int() == 1:
                                has_solution = True
                        elif (
                            hasattr(r.result, "model_checking_solution")
                            and r.result.model_checking_solution is not None
                        ):
                            if r.result.model_checking_solution.status.to_int() == 1:
                                has_solution = True
                    self.sem.release()
                    if has_solution:
                        break

        return sample
