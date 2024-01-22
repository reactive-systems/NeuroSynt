import json

from ml2.ltl import DecompLTLSpec

from .argument_parser import ArgumentParser
from .neurosynt_tool import NeuroSynt


def read_spec(spec_file, config) -> DecompLTLSpec:
    syfco = (
        config["syfco"]["class"](**config["syfco"]["service_args"])
        if spec_file.endswith(".tlsf")
        else None
    )

    if syfco is not None:
        res = syfco.from_tlsf_file(spec_file)
        del syfco
        return res
    else:
        return DecompLTLSpec.from_bosy_file(spec_file)


def setup_tools(config):
    symbolic_solver = config["symbolic_solver"]["class"](
        **config["symbolic_solver"]["service_args"]
    )

    model_checker = config["model_checker"]["class"](**config["model_checker"]["service_args"])

    config["neural_solver"]["service_args"]["mc_port"] = model_checker.port
    config["neural_solver"]["service_args"]["verifier"] = model_checker.__class__.__name__
    neural_solver = config["neural_solver"]["class"](**config["neural_solver"]["service_args"])
    return symbolic_solver, model_checker, neural_solver


def write_output(output_file, spec, solution):
    if output_file:
        with open(output_file, "w") as f:
            data = {"specification": spec.to_dict(), "solutions": []}
            for res in solution.results:
                if res.name == "symbolic_solver":
                    solution = {
                        "tool": res.name,
                        "time": res.time,
                        "status": res.result.status.token(),
                        "solution": str(res.result.system),
                    }
                elif res.name == "neural_solver":
                    assert hasattr(res.result, "model_checking_solution")
                    assert hasattr(res.result, "synthesis_solution")
                    solution = {
                        "tool": res.name,
                        "time": res.time,
                        "status": res.result.synthesis_solution.status.token(),
                        "solution": str(res.result.synthesis_solution.system),
                        "model_checking_status": res.result.model_checking_solution.status.token(),
                    }
                else:
                    raise ValueError("Cannot serialize solver result")
                data["solutions"].append(solution)
            json.dump(data, f, indent=4)
    else:
        for res in solution.results:
            if res.name == "symbolic_solver":
                print(res.name)
                print(res.time)
                print(res.result.status.token().upper())
                print(res.result.system.to_str())
            elif res.name == "neural_solver":
                assert hasattr(res.result, "model_checking_solution")
                assert res.result.model_checking_solution is not None
                assert hasattr(res.result, "synthesis_solution")
                assert res.result.synthesis_solution is not None
                print(res.name)
                print(res.time)
                print(res.result.model_checking_solution.status.token().upper())
                print(res.result.synthesis_solution.status.token().upper())
                print(res.result.synthesis_solution.system.to_str())
            else:
                raise ValueError("Cannot print solver result")


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parser.parse_args()
    config = parser.parse_config(args)

    print(config)

    for tool_kind, values in config.items():
        if "class" not in values or "service_args" not in values:
            raise Exception("config not formatted properly\n" + str(values))

    tool = NeuroSynt()
    symbolic_solver, model_checker, neural_solver = setup_tools(config)

    tools = {
        "symbolic_solver": (symbolic_solver, config["symbolic_solver"]["tool_run_args"]),
        "neural_solver": (neural_solver, config["neural_solver"]["tool_run_args"]),
    }

    if args.subparser == "benchmark":
        dataset, save_as, auto_version, upload, add_to_wandb, sample = (
            args.dataset,
            args.save_as,
            args.auto_version,
            args.upload,
            args.add_to_wandb,
            args.sample,
        )
        tool.benchmark(
            tools=tools,
            config=config,
            dataset=dataset,
            output=save_as,
            auto_version=auto_version,
            upload=upload,
            add_to_wandb=add_to_wandb,
            sample=sample,
        )

    elif args.subparser == "synthesize":
        spec_file, output_file, all_results = args.spec, args.output, args.all_results

        spec = read_spec(spec_file, config)

        solution = tool.synthesize(
            spec,
            tools,
            all_results,
        )
        write_output(output_file, spec, solution)
    else:
        raise Exception("Cannot handle subparser")

    del symbolic_solver
    del model_checker
    del neural_solver
