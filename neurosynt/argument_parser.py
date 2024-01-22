import argparse
import os
import textwrap
from argparse import RawTextHelpFormatter

import yaml
from ml2.tools.bosy import BoSy
from ml2.tools.ltl_tool.generic_model_checker import GenericModelChecker
from ml2.tools.ltl_tool.generic_synthesis_tool import GenericSynthesisTool
from ml2.tools.neurosynt import NeuroSynt as ML2Solver
from ml2.tools.nuxmv import NuxmvMC
from ml2.tools.spot import SpotAIGERMC
from ml2.tools.strix import Strix
from ml2.tools.syfco import Syfco


class ArgumentParser:
    default_configurations = {
        "symbolic_solver": {
            "strix": {
                "class": Strix,
                "tool_run_args": {"timeout": 120, "--compression": "more"},
                "service_args": {
                    "cpu_count": 2,
                    "mem_limit": "2g",
                    "port": None,
                    "start_containerized_service": True,
                    "start_service": False,
                    "start_up_timeout": 60,
                    "channel_ready_timeout": 15,
                },
            },
            "bosy": {
                "class": BoSy,
                "tool_run_args": {
                    "timeout": 120,
                    "--optimize": "",
                },
                "service_args": {
                    "cpu_count": 2,
                    "mem_limit": "2g",
                    "port": None,
                    "start_containerized_service": True,
                    "start_service": False,
                    "start_up_timeout": 60,
                    "channel_ready_timeout": 15,
                },
            },
            "spot": {
                "class": SpotAIGERMC,
                "tool_run_args": {
                    "timeout": 120,
                },
                "service_args": {
                    "cpu_count": 2,
                    "mem_limit": "2g",
                    "port": None,
                    "start_containerized_service": True,
                    "start_service": False,
                    "start_up_timeout": 60,
                    "channel_ready_timeout": 15,
                },
            },
        },
        "model_checker": {
            "nuxmv": {
                "class": NuxmvMC,
                "tool_run_args": {
                    "timeout": 10,
                },
                "service_args": {
                    "cpu_count": 1,
                    "mem_limit": "2g",
                    "port": None,
                    "start_containerized_service": True,
                    "start_service": False,
                    "start_up_timeout": 60,
                    "channel_ready_timeout": 15,
                },
            },
            "spot": {
                "class": SpotAIGERMC,
                "tool_run_args": {
                    "timeout": 10,
                },
                "service_args": {
                    "cpu_count": 1,
                    "mem_limit": "2g",
                    "port": None,
                    "start_containerized_service": True,
                    "start_service": False,
                    "start_up_timeout": 60,
                    "channel_ready_timeout": 15,
                },
            },
        },
        "neural_solver": {
            "ml2solver": {
                "class": ML2Solver,
                "tool_run_args": {},
                "service_args": {
                    "nvidia_gpus": True,
                    "mem_limit": "10g",
                    "start_containerized_service": True,
                    "start_service": False,
                },
                "tool_setup_args": {
                    "batch_size": 32,
                    "alpha": 0.5,
                    "num_properties": 18,
                    "length_properties": 40,
                    "beam_size": 2,
                    "check_setup": True,
                    "model": "ht-3",
                },
            },
        },
    }

    syfco_default = {
        "class": Syfco,
        "tool_run_args": {},
        "service_args": {
            "cpu_count": 1,
            "mem_limit": "2g",
            "port": None,
            "start_containerized_service": True,
            "start_service": False,
            "start_up_timeout": 60,
            "channel_ready_timeout": 15,
        },
    }

    def __init__(self):
        config_example = textwrap.dedent(
            """\
Yaml example file for config:
    symbolic_solver:
        tool: strix
        tool_run_args:
            "timeout": 120
            "--threads": 4
            "--minimize": ""
            "--auto": ""
        service_args:
            "cpu_count": 12
            "mem_limit": "2g"
            "port": 50052
            "start_containerized_service": True
            "start_service": False
            "start_up_timeout": 60
            "channel_ready_timeout": 15
    model_checker:
        tool: 1234
        tool_run_args:
            "timeout": 10
        service_args:
            "cpu_count": 1
            "mem_limit": "2g"
            "start_containerized_service": False
            "start_service": False
            "start_up_timeout": 60
            "channel_ready_timeout": 15
        tool_setup_args:
            "foo": "bar"
    neural_solver:
        tool: ml2solver
        service_args:
            "nvidia_gpus": True
            "mem_limit": "10g"
            "start_containerized_service": True
            "start_service": False
        tool_setup_args:
            "batch_size": 32
            "alpha": 0.5
            "num_properties": 18
            "length_properties": 40
            "beam_size": 2
            "check_setup": True
            "model": "ht-3"
"""
        )
        description = textwrap.dedent(
            """\
NeuroSynt:
    Choose synthesize for syhthesizing a single specification (synthesize -h for help)
    Choose benchmark to run the benchmark setting (benchmark -h for help)
"""
        )
        # TODO better description
        self.parser = argparse.ArgumentParser(
            formatter_class=RawTextHelpFormatter,
            prog="NeuroSynt",
            description=description,
        )
        syn_description = textwrap.dedent(
            """\
NeuroSynt:
    Set the specification to synthesize with --spec.
    Set a config file for configuration parameters of the tools with --config or by setting the remaining arguments.
    Set an output file with --output instead of writing to stdout.
    Set --all-results if incorrect results (of the neural solver) should be displayed as well.
"""
        )

        subparsers = self.parser.add_subparsers(dest="subparser")
        self.syn_parser = subparsers.add_parser(
            "synthesize",
            description=syn_description,
            formatter_class=RawTextHelpFormatter,
            epilog=config_example,
            help="synthesize a single specification",
        )
        self.add_syn_parser_args(parser=self.syn_parser)

        benchmark_description = textwrap.dedent(
            """\
NeuroSynt:
    Set the name of the dataset to benchmark with --dataset.
    Set a config file for configuration parameters of the tools with --config or by setting the remaining arguments.
    Set an output folder with --save-as.
"""
        )
        self.benchmark_parser = subparsers.add_parser(
            "benchmark",
            description=benchmark_description,
            formatter_class=RawTextHelpFormatter,
            epilog=config_example,
            help="run the benchmark setting",
        )
        self.add_benchmark_parser_args(parser=self.benchmark_parser)

    def fill_from_default(self, config, tool_kind: str, tool: str):
        if tool in self.default_configurations[tool_kind]:
            config[tool_kind] = self.default_configurations[tool_kind][tool]
        else:
            if isinstance(tool, int) or tool.isdigit():  # resembles a port
                config[tool_kind] = {
                    "tool_run_args": {},
                    "tool_setup_args": {},
                    "service_args": {
                        "port": int(tool),
                        "start_service": False,
                        "start_containerized_service": False,
                    },
                }
            else:  # should resemble docker image
                config[tool_kind] = {
                    "tool_run_args": {},
                    "tool_setup_args": {},
                    "service_args": {
                        "image": tool,
                        "start_service": False,
                        "start_containerized_service": True,
                    },
                }
            if tool_kind == "symbolic_solver":
                config[tool_kind]["class"] = GenericSynthesisTool
            elif tool_kind == "model_checker":
                config[tool_kind]["class"] = GenericModelChecker

    def update_with_cli_config(self, config, tool_kind, tool, cli_config):
        if tool_kind in cli_config and tool == cli_config[tool_kind]["tool"]:
            for arg_kind in config[tool_kind]:
                if arg_kind == "class":
                    continue
                if arg_kind in cli_config[tool_kind]:
                    for arg_key, arg_value in cli_config[tool_kind][arg_kind].items():
                        config[tool_kind][arg_kind][arg_key] = arg_value

    def update_with_cli_args(self, config, tool_kind, args, arg_kind, args_suffix):
        if args[tool_kind + args_suffix] is not None:
            for arg in args[tool_kind + args_suffix]:
                key, value = arg.split("=") if len(arg.split("=")) == 2 else (arg, True)
                config[tool_kind][arg_kind][key] = value

    def parse_config(self, args):
        if args.config is not None:
            cli_config = self.load_config_file(args.config)
        else:
            cli_config = None

        config = {}
        for tool_kind in self.default_configurations:
            tool = (
                vars(args)[tool_kind]
                or (
                    cli_config[tool_kind]["tool"]
                    if (cli_config is not None and tool_kind in cli_config)
                    else None
                )
                or list(self.default_configurations[tool_kind].keys())[0]
            )
            self.fill_from_default(config, tool_kind, tool)
            self.update_with_cli_config(config, tool_kind, tool, cli_config)
            self.update_with_cli_args(config, tool_kind, vars(args), "tool_run_args", "_run_args")
            self.update_with_cli_args(
                config, tool_kind, vars(args), "service_args", "_service_args"
            )
            if "tool_setup_args" in config[tool_kind]:
                self.update_with_cli_args(
                    config, tool_kind, vars(args), "tool_setup_args", "_setup_args"
                )
                config[tool_kind]["service_args"]["setup_parameters"] = config[tool_kind].pop(
                    "tool_setup_args"
                )
        config["syfco"] = self.syfco_default
        self.update_with_cli_args(config, "syfco", vars(args), "tool_run_args", "_run_args")

        return config

    def load_config_file(self, path):
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def add_syn_parser_args(self, parser):
        parser.add_argument(
            "--spec",
            type=self.ltl_or_tlsf_file,
            required=True,
            help="Path to a .ltl or .tlsf file containing the specification",
        )
        parser.add_argument(
            "--config",
            type=self.yaml_file,
            help="Path to a .yaml file containing the configuration options. CLI arguments override values in the config.",
        )

        parser.add_argument(
            "--output",
            type=str,
            help="Set the path and filename for the output file where results will be saved. If not specified, results will be printed to the console.",
        )

        parser.add_argument(
            "--all-results",
            dest="all_results",
            action="store_true",
            help="Set if even incorrect results from the neural solver should be displayed",
        )

        self.add_symbolic_solver_args(parser=parser)
        self.add_neural_solver_args(parser=parser)
        self.add_model_checker_args(parser=parser)
        self.add_syfco_args(parser=parser)

    def add_benchmark_parser_args(self, parser):
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Name of the dataset to benchmark",
        )
        parser.add_argument(
            "--config",
            type=self.yaml_file,
            help="Path to a .yaml file containing the configuration options. CLI arguments override values in the config.",
        )

        parser.add_argument(
            "--save-as",
            dest="save_as",
            type=str,
            required=True,
            help="Set the name for the folder, the results will be saved in. Subfolder of ~/ml2-storage/ltl-syn",
        )

        parser.add_argument(
            "--auto-version",
            dest="auto_version",
            action="store_true",
            help="Set to auto-version the selected save-as folder",
        )

        parser.add_argument(
            "--upload",
            action="store_true",
            help="Set to upload the benchmarking results to google cloud",
        )

        parser.add_argument(
            "--add-to-wandb",
            dest="add_to_wandb",
            action="store_true",
            help="Set if the benchmarking run should be added to Weights & Biases",
        )

        parser.add_argument(
            "--sample",
            dest="sample",
            type=int,
            required=False,
            help="Set the the number of samples to be evaluated.",
        )

        self.add_symbolic_solver_args(parser=parser)
        self.add_neural_solver_args(parser=parser)
        self.add_model_checker_args(parser=parser)
        self.add_syfco_args(parser=parser)

    def ltl_or_tlsf_file(self, path):
        _, ext = os.path.splitext(path)
        if ext.lower() not in (".ltl", ".tlsf", ".json"):
            raise argparse.ArgumentTypeError(
                "The specification file must be a bosy file (.ltl or .json) or a .tlsf file"
            )
        return path

    def yaml_file(self, path):
        _, ext = os.path.splitext(path)
        if ext.lower() not in (".yml", ".yaml"):
            raise argparse.ArgumentTypeError("The config file file must be a .yml or .yaml file")
        return path

    def add_symbolic_solver_args(self, parser):
        parser.add_argument(
            "--symbolic-solver",
            type=str,
            help="Name or grpc port or docker container of the symbolic solver",
        )
        parser.add_argument(
            "--symbolic-solver-run-args",
            nargs="+",
            help='Specifying the arguments required by the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--symbolic-solver-setup-args",
            nargs="+",
            help='Specifying the arguments for the setup of the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--symbolic-solver-service-args",
            nargs="+",
            help='Specifying the arguments for the service setup: "--arg1 --arg2=val2 arg3=val3 ..."',
        )

    def add_neural_solver_args(self, parser):
        parser.add_argument(
            "--neural-solver",
            type=str,
            help="Name or grpc port or docker container of the neural solver",
        )
        parser.add_argument(
            "--neural-solver-run-args",
            nargs="+",
            help='Specifying the arguments required by the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--neural-solver-setup-args",
            nargs="+",
            help='Specifying the arguments for the setup of the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--neural-solver-service-args",
            nargs="+",
            help='Specifying the arguments for the service setup: "--arg1 --arg2=val2 arg3=val3 ..."',
        )

    def add_model_checker_args(self, parser):
        parser.add_argument(
            "--model-checker",
            type=str,
            help="Name or grpc port or docker container of the model checker for the neural solver",
        )
        parser.add_argument(
            "--model-checker-run-args",
            nargs="+",
            help='Specifying the arguments required by the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--model-checker-setup-args",
            nargs="+",
            help='Specifying the arguments for the setup of the solver: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--model-checker-service-args",
            nargs="+",
            help='Specifying the arguments for the service setup: "--arg1 --arg2=val2 arg3=val3 ..."',
        )

    def add_syfco_args(self, parser):
        parser.add_argument(
            "--syfco-run-args",
            help='Specifying the arguments required by syfco: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
        parser.add_argument(
            "--syfco-setup-args",
            help='Specifying the arguments for the setup of syfco: "--arg1 --arg2=val2 arg3=val3 ..."',
        )
