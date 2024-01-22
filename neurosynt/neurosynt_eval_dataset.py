"""CSV dataset"""

import logging
import zipfile
from copy import deepcopy
from functools import cmp_to_key
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import requests
from ml2.aiger import AIGERCircuit
from ml2.datasets import CSVDataset
from ml2.dtypes import CSVDict
from ml2.ltl import DecompLTLSpec
from ml2.ltl.ltl_spec import LTLSpecDataset
from ml2.ltl.ltl_syn import LTLSynEvalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPEC_STATS = ["size", "num_properties", "max_prop_length", "num_aps"]
CIRC_STATS = ["num_latches", "num_ands", "max_var_id"]


class NeuroSyntEvalDataset(CSVDataset[CSVDict]):
    def __init__(
        self,
        dtype: Type[CSVDict],
        df: pd.DataFrame = None,
        add_stats: bool = True,
        **kwargs,
    ):
        df.reset_index(drop=True, inplace=True)
        super().__init__(dtype=dtype, df=df, **kwargs)
        if add_stats:
            self.df["result_model_checking_satisfied"] = self.df[
                "result_model_checking_satisfied"
            ].apply(lambda x: (x == "1" or x) if x != "" and not pd.isnull(x) else np.nan)
            if "result_valid" not in self.df.columns:
                self.add_valid()
            if (
                "result_num_latches" not in self.df.columns
                or "result_num_ands" not in self.df.columns
            ) and "result_circuit" in self.df.columns:
                self.df = self._add_circ_stats(self.df, prefix="result_")
            if (
                "result_synthesis_num_latches" not in self.df.columns
                or "result_synthesis_num_ands" not in self.df.columns
            ) and "result_synthesis_circuit" in self.df.columns:
                self.add_circ_stats()
            if (
                (
                    "input_size" not in self.df.columns
                    or "input_num_properties" not in self.df.columns
                    or "input_num_aps" not in self.df.columns
                    or "input_max_prop_length" not in self.df.columns
                )
                and "input_assumptions" in self.df.columns
                and "input_guarantees" in self.df.columns
            ):
                self.add_spec_stats()
            self.add_par_time()

    def add_par_time(self):
        self.df["result_duration_par"] = self.df.apply(
            lambda row: row["result_duration"]
            if (
                "result_synthesis_duration" not in row.index
                or "result_model_checking_duration" not in row.index
                or pd.isnull(row["result_synthesis_duration"])
                or pd.isnull(row["result_model_checking_duration"])
                or row["result_synthesis_duration"] == ""
                or row["result_model_checking_duration"] == ""
            )
            else (
                float(row["result_synthesis_duration"])
                + float(row["result_model_checking_duration"])
            ),
            axis=1,
        )
        self.df["result_duration_par"] = self.df["result_duration_par"].apply(
            lambda x: np.nan if pd.isnull(x) or x == "" else float(x)
        )

    @staticmethod
    def _add_spec_stats(df: pd.DataFrame, prefix: str = "", suffix: str = ""):
        cols = [prefix + e + suffix for e in SPEC_STATS]

        def add_spec_stats_row(row, prefix: str, suffix: str):
            spec: DecompLTLSpec = DecompLTLSpec.from_csv_fields(
                row.to_dict(), prefix=prefix, suffix=suffix
            )
            try:
                max_prop_length = 0
                size = len(spec.guarantees) - 1 + max(len(spec.assumptions) - 1, 0)
                [size := size + g.size() for g in spec.guarantees]
                [max_prop_length := max(max_prop_length, g.size()) for g in spec.guarantees]
                if len(spec.assumptions) != 0:
                    [size := size + a.size() for a in spec.assumptions]
                    [max_prop_length := max(max_prop_length, a.size()) for a in spec.assumptions]
                    size += 1
            except RecursionError:
                size = "inf"
                max_prop_length = "inf"

            return {
                "spec_size": size,
                "num_properties": len(spec.guarantees) + len(spec.assumptions),
                "max_prop_length": max_prop_length,
                "num_aps": len(spec.inputs) + len(spec.outputs),
            }

        df[cols] = df.apply(
            lambda row: add_spec_stats_row(row, prefix=prefix, suffix=suffix),
            axis=1,
            result_type="expand",
        )
        return df

    def add_spec_stats(self):
        self.df = self._add_spec_stats(self.df, prefix="input_")

    @staticmethod
    def _add_circ_stats(df: pd.DataFrame, prefix: str = "", suffix: str = ""):
        cols = [prefix + e + suffix for e in CIRC_STATS]

        def add_circ_stats_row(row, prefix: str, suffix: str):
            circ: AIGERCircuit = AIGERCircuit.from_csv_fields(
                row.to_dict(), prefix=prefix, suffix=suffix
            )
            if circ is not None:
                return {
                    "num_latches": circ.num_latches,
                    "num_ands": circ.num_ands,
                    "max_var_id": circ.max_var_id,
                }
            else:
                return {}

        df[cols] = df.apply(
            lambda row: add_circ_stats_row(row, prefix=prefix, suffix=suffix),
            axis=1,
            result_type="expand",
        )
        return df

    def add_circ_stats(self):
        self.df = self._add_circ_stats(self.df, prefix="result_")
        self.df = self._add_circ_stats(self.df, prefix="result_synthesis_")

    def add_valid(self):
        def add_valid_row(row):
            if not pd.isnull(row["result_model_checking_satisfied"]):
                return row["result_model_checking_satisfied"]
            else:
                return row["result_realizable"] == "1" or row["result_realizable"] == "0"

        self.df["result_valid"] = self.df.apply(lambda row: add_valid_row(row), axis=1)

    @staticmethod
    def get_latches(row):
        if "result_synthesis_num_latches" in row.index and not pd.isnull(
            row["result_synthesis_num_latches"]
        ):
            return row["result_synthesis_num_latches"]
        if "result_num_latches" in row.index and not pd.isnull(row["result_num_latches"]):
            return row["result_num_latches"]
        return pd.NA

    @staticmethod
    def get_ands(row):
        if "result_synthesis_num_ands" in row.index and not pd.isnull(
            row["result_synthesis_num_ands"]
        ):
            return row["result_synthesis_num_ands"]
        if "result_num_ands" in row.index and not pd.isnull(row["result_num_ands"]):
            return row["result_num_ands"]
        return pd.NA

    def _select_rows(
        self,
        smallest: bool,
        fastest: bool,
        inplace: bool,
        realizable_only: bool = False,
        par: bool = True,
    ):
        def compare(row1, row2):
            dif = self.get_latches(row1) - self.get_latches(row2)
            if dif != 0:
                return dif
            else:
                return self.get_ands(row1) - self.get_ands(row2)

        def select_row(df, fastest: bool, smallest: bool) -> int:
            if (fastest and smallest) or (not smallest and not fastest):
                raise ValueError
            df_v = df[df["result_valid"]]
            if len(df_v) > 1:
                if fastest:
                    if par:
                        return df_v["result_duration_par"].idxmin()
                    else:
                        return df_v["result_duration"].idxmin()
                elif smallest:
                    s_list = sorted((row for _, row in df_v.iterrows()), key=cmp_to_key(compare))
                    return s_list[0].name
            elif len(df_v) == 1:
                return df_v.index.to_list()[0]
            else:
                return df.first_valid_index()

        if inplace:
            obj = self
        else:
            obj = deepcopy(self)

        if realizable_only:
            obj.df = obj.df[
                (obj.df["result_synthesis_realizable"] == 1)
                | (obj.df["result_realizable"] == 1)
                | (obj.df["result_synthesis_realizable"] == "1")
                | (obj.df["result_realizable"] == "1")
            ]

        new_df = obj.df.loc[
            [
                select_row(row, fastest=fastest, smallest=smallest)
                for row in (
                    (obj.df.loc[y]) for _, y in obj.df.groupby("input_name").groups.items()
                )
            ]
        ]

        obj.df = new_df

        return obj

    def group_agg_fastest(self, inplace: bool, realizable_only: bool = False, par: bool = True):
        return self._select_rows(
            smallest=False, fastest=True, inplace=inplace, realizable_only=realizable_only, par=par
        )

    def group_agg_smallest(self, inplace: bool, realizable_only: bool = False):
        return self._select_rows(
            smallest=True, fastest=False, inplace=inplace, realizable_only=realizable_only
        )

    def solved_by(self, tools: List[str]):
        def all_valid(row):
            if row["result_tool"] in tools and row["result_valid"]:
                df = self.get(row)
                df_f = df[df["result_tool"].isin(tools)]
                return len(tools) == len(df_f[df_f["result_valid"]])
            return False

        return self.df[[all_valid(y) for _, y in self.df.iterrows()]]

    def fastest_samples(
        self,
        by_tool: Optional[str] = None,
        out_of_tools: Optional[List[str]] = None,
        include_invalid: bool = False,
        parallel_time: bool = True,
        realizable_only=False,
    ) -> pd.DataFrame:
        copy = deepcopy(self)
        if out_of_tools is not None:
            copy.df = copy.df[copy.df["result_tool"].isin(out_of_tools)]
            ref = deepcopy(copy.df)
        else:
            ref = self.df
        copy.group_agg_fastest(inplace=True, par=parallel_time, realizable_only=realizable_only)
        if by_tool is not None:
            copy.df = copy.df[copy.df["result_tool"] == by_tool]
        if not include_invalid:
            copy.df = copy.df[copy.df["result_valid"]]

        def f(row):
            ref_row = ref[
                (ref["input_name"] == row["input_name"])
                & (ref["result_tool"] != row["result_tool"])
            ].iloc[0]
            if parallel_time:
                return (
                    (ref_row["result_duration_par"] - row["result_duration_par"])
                    if ref_row["result_valid"]
                    else np.nan
                )
            else:
                return (
                    (ref_row["result_duration"] - row["result_duration"])
                    if ref_row["result_valid"]
                    else np.nan
                )

        if len(ref["result_tool"].value_counts()) == 2:
            copy.df["result_duration_diff"] = copy.df.apply(f, axis=1)
        return copy.df

    def compare_smaller(
        self,
        by_tool: str,
        out_of_tools: Optional[List[str]] = None,
        realizable_only: bool = False,
    ) -> pd.Series:
        """Compares the differences in size of multiple evaluations

        Args:
            by_tool (str):The tool which the comparisons are relative too
            out_of_tools (Optional[List[str]], optional): Collection of tools to which is compared. Defaults to None, meaning all tools.

        Returns:
            pd.Series[Union[float,bool]]: Statistics of the comparison.
            The average of latches of all samples by by_tool, the average of latches of all samples not by by_tool but in out_of_tools, the average of latches of all samples in out_of_tools.
            Whether the average of latches of all samples by by_tool is smaller than the average of latches of all samples in out_of_tools,
            How much percent this is smaller or larger  than the out_of_tools average,
            How much percent this is smaller or larger than the out_of_tools but not by_tools average,
        """
        if out_of_tools is None:
            out_of_tools = list(self.df["result_tool"].value_counts().index)

        solved = NeuroSyntEvalDataset(
            df=deepcopy(self.solved_by(out_of_tools)), name="all_solved", dtype=CSVDict
        )

        if realizable_only:
            solved.df = solved.df[
                (solved.df["result_synthesis_realizable"] == 1)
                | (solved.df["result_realizable"] == 1)
                | (solved.df["result_synthesis_realizable"] == "1")
                | (solved.df["result_realizable"] == "1")
            ]

        solved.df["latches"] = solved.df.apply(self.get_latches, axis=1)

        avg = solved.df["latches"].mean()
        tool_avg = solved.df[solved.df["result_tool"] == by_tool]["latches"].mean()
        not_tool_avg = solved.df[solved.df["result_tool"] != by_tool]["latches"].mean()

        return pd.Series(
            {
                "by_tool average": tool_avg,
                "out_of_tools average": avg,
                "out_of_tools but not by_tools average": not_tool_avg,
                "smaller": tool_avg < not_tool_avg,
                "% smaller or larger than out_of_tools": (((avg - tool_avg) / avg) * 100)
                if tool_avg < not_tool_avg
                else (((not_tool_avg - tool_avg) / not_tool_avg) * 100),
                "% smaller or larger than out_of_tools but not by_tools average": (
                    ((not_tool_avg - tool_avg) / not_tool_avg) * 100
                )
                if tool_avg < not_tool_avg
                else (((tool_avg - not_tool_avg) / not_tool_avg) * 100 - 100),
            }
        )

    def exclusively_solved(
        self,
        by_tool: str,
        out_of_tools: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        copy = deepcopy(self)
        if out_of_tools is not None:
            copy.df = copy.df[copy.df["result_tool"].isin(out_of_tools)]

        tool_df = copy.df[(copy.df["result_tool"] == by_tool) & copy.df["result_valid"]]
        return tool_df[
            [
                (
                    len(
                        copy.df[
                            (copy.df["input_name"] == y["input_name"]) & copy.df["result_valid"]
                        ]
                    )
                    == 1
                )
                for _, y in tool_df.iterrows()
            ]
        ]

    def smallest_samples(
        self,
        by_tool: Optional[str] = None,
        out_of_tools: Optional[List[str]] = None,
        realizable_only: bool = False,
        include_invalid: bool = False,
    ) -> pd.DataFrame:
        copy = deepcopy(self)
        if out_of_tools is not None:
            copy.df = copy.df[copy.df["result_tool"].isin(out_of_tools)]
            ref = deepcopy(copy.df)
        else:
            ref = self.df
        copy.group_agg_smallest(inplace=True, realizable_only=realizable_only)
        if by_tool is not None:
            copy.df = copy.df[copy.df["result_tool"] == by_tool]
        if not include_invalid:
            copy.df = copy.df[copy.df["result_valid"]]

        def f1(row):
            ref = self.df[
                (self.df["input_name"] == row["input_name"])
                & (self.df["result_tool"] != row["result_tool"])
            ].iloc[0]
            return (
                (self.get_latches(ref) - self.get_latches(row)) if ref["result_valid"] else np.nan
            )

        def f2(row):
            ref = self.df[
                (self.df["input_name"] == row["input_name"])
                & (self.df["result_tool"] != row["result_tool"])
            ].iloc[0]
            return (self.get_ands(ref) - self.get_ands(row)) if ref["result_valid"] else np.nan

        if len(ref["result_tool"].value_counts()) == 2:
            copy.df["result_num_latches_diff"] = copy.df.apply(f1, axis=1)
            copy.df["result_num_ands_diff"] = copy.df.apply(f2, axis=1)

        def check_strict(row) -> bool:
            df = ref[ref["input_name"] == row["input_name"]]
            df["latches"] = df.apply(self.get_latches, axis=1)
            df["ands"] = df.apply(self.get_ands, axis=1)
            return (
                len(
                    df[
                        (df["latches"] == self.get_latches(row))
                        & (df["ands"] == self.get_ands(row))
                    ]["result_tool"].value_counts()
                )
                == 1
            )

        copy.df["strict"] = copy.df.apply(check_strict, axis=1)
        return copy.df

    def get(
        self,
        reference: Union[str, pd.Series, pd.DataFrame],
        tools: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Gets the DataFrame containing results for all solvers (or a given solver) with the given unique references.

        Args:
            reference (Union[str, pd.Series, pd.DataFrame]): Either the unique name of the sample (column input_name) or the row of the sample (i.e. Series) or a DataFrame of multiple samples.
            tools (Optional[List[str]], optional): Only show results of the given tools. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe consisting of the requested samples
        """
        if isinstance(reference, str):
            df = self.df[self.df["input_name"] == reference]
        elif isinstance(reference, pd.Series):
            df = self.df[self.df["input_name"].isin([reference["input_name"]])]
        elif isinstance(reference, pd.DataFrame):
            df = self.df[self.df["input_name"].isin(list(reference["input_name"]))]

        if tools is None:
            return df
        else:
            return df[df["result_tool"].isin(tools)]

    @classmethod
    def from_syntcomp(
        cls,
        url: Optional[str] = None,
        reference: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> "NeuroSyntEvalDataset":
        # TODO this takes an unbelievably long time. Not sure why.
        assert "df" not in kwargs
        if reference is None:
            reference = "ltl-spec/sc-1"
        if url is None:
            url = "https://www.starexec.org/starexec/secure/download?token=21966710&type=job&id=53822&returnids=true&getcompleted=false"

        syntcomp_df = cls.download_syntcomp(url, filename)

        specs: LTLSpecDataset = LTLSpecDataset.load("ltl-spec/sc-1")  # type: ignore
        specs_df = specs.df.rename(lambda name: "input_" + name, axis=1)

        df = cls.convert_from_syntcomp(syntcomp_df)
        df = df.merge(right=specs_df, on="input_name")
        obj = cls(df=df, **kwargs)
        obj.df = obj.df.reindex()
        obj.add_spec_stats()
        obj.metadata["dataset"] = specs.bucket_path
        obj.metadata["syntcomp_url"] = url
        return obj

    @staticmethod
    def download_syntcomp(url: Optional[str] = None, filename: Optional[str] = None):
        if url is None:
            url = "https://www.starexec.org/starexec/secure/download?token=21966710&type=job&id=53822&returnids=true&getcompleted=false"
        if filename is None:
            u = requests.get(url, timeout=120)
            input_zip = zipfile.ZipFile(BytesIO(u.content))
        else:
            input_zip = zipfile.ZipFile(filename)
        return pd.read_csv(StringIO(str(input_zip.read(input_zip.namelist()[0]), "utf-8")))

    @staticmethod
    def convert_from_syntcomp(df: pd.DataFrame) -> pd.DataFrame:
        def convert_status(row):
            if row["result"] == "REALIZABLE" or row["result"] == "NEW-REALIZABLE":
                return 1
            elif row["result"] == "UNREALIZABLE" or row["result"] == "NEW-UNREALIZABLE":
                return 0
            elif row["status"].startswith("timeout"):
                return -2
            else:
                return -1

        def convert_detailed_status(row):
            if row["result"] == "REALIZABLE" or row["result"] == "NEW-REALIZABLE":
                return "REALIZABLE"
            elif row["result"] == "UNREALIZABLE" or row["result"] == "NEW-UNREALIZABLE":
                return "UNREALIZABLE"
            else:
                return row["status"] + ", " + row["Error"]

        ret = deepcopy(df)
        ret["input_name"] = df.apply(
            lambda row: row["benchmark"].split("/")[1].split(".")[0], axis=1
        )

        ret["result_tool"] = df.apply(
            lambda row: row["solver"] + "-" + row["configuration"], axis=1
        )
        ret["result_synthesis_num_latches"] = pd.to_numeric(
            df.apply(
                lambda row: row["Synthesis_latches"]
                if row["Synthesis_latches"] != "-"
                else np.nan,
                axis=1,
            )
        )
        ret["result_synthesis_num_ands"] = pd.to_numeric(
            df.apply(
                lambda row: row["Synthesis_gates"] if row["Synthesis_gates"] != "-" else np.nan,
                axis=1,
            )
        )
        ret["result_valid"] = df.apply(
            lambda row: row["Model_check_result"] == "SUCCESS"
            or row["Expected_result"].upper() == row["result"].upper()
            or row["result"]
            == "NEW-UNREALIZABLE",  # We assume correctness of the solver for NEW-UNREALIZABLE, as there is no possibility of model checking
            axis=1,
        )
        ret["result_model_checking_satisfied"] = df.apply(
            lambda row: row["Model_check_result"] == "SUCCESS", axis=1
        )
        ret["result_model_checking_detailed_status"] = df.apply(
            lambda row: "SATISFIED" if row["Model_check_result"] == "SUCCESS" else "", axis=1
        )
        ret["result_duration"] = pd.to_numeric(df.apply(lambda row: row["wallclock time"], axis=1))
        ret["result_synthesis_realizable"] = df.apply(convert_status, axis=1)
        ret["result_synthesis_detailed_status"] = df.apply(convert_detailed_status, axis=1)
        ret["result_detailed_status"] = np.nan
        ret["result_realizable"] = np.nan
        ret["result_num_latches"] = np.nan
        ret["result_num_ands"] = np.nan

        ret = ret.drop(
            [
                "pair id",
                "benchmark",
                "benchmark id",
                "solver",
                "solver id",
                "configuration",
                "configuration id",
                "status",
                "cpu time",
                "wallclock time",
                "memory usage",
                "result",
                "Model_check_result",
                "Output_by_reference",
                "Synthesis_latches",
                "Error",
                "Difference_to_reference",
                "Expected_result",
                "Synthesis_gates",
                "unknown",
            ],
            axis=1,
        )
        return ret

    @classmethod
    def from_merge(
        cls, evaluations: List["NeuroSyntEvalDataset"], **kwargs
    ) -> "NeuroSyntEvalDataset":
        comparisons = []
        for el1 in evaluations:
            for el2 in evaluations:
                comparisons.append(
                    len(
                        set(el1.df["result_tool"].value_counts().index)
                        and set(el2.df["result_tool"].value_counts().index)
                    )
                    == 0
                )

        if any(comparisons):
            raise ValueError(
                "When merging multiple portfolio evaluations, all tools must be disjunct"
            )

        df = pd.concat([eval.df for eval in evaluations])
        metadata = {
            "joined_from": [{**eval.metadata, "name": eval.bucket_path} for eval in evaluations]
        }

        return cls(dtype=CSVDict, df=df, metadata=metadata, add_stats=False, **kwargs)

    def collapse(
        self,
        collapse_dict: Dict[str, List],
        fastest: bool = False,
        smallest: bool = False,
        inplace: bool = False,
        realizable_only=False,
    ):
        if (fastest and smallest) or (not smallest and not fastest):
            raise ValueError

        tools = []
        evaluations = []
        for k, v in collapse_dict.items():
            tools = tools + v
            if fastest:
                df = self.fastest_samples(
                    out_of_tools=v,
                    include_invalid=True,
                    realizable_only=realizable_only,
                )
            else:
                df = self.smallest_samples(
                    out_of_tools=v,
                    include_invalid=True,
                    realizable_only=realizable_only,
                )
            df["result_tool"] = k
            evaluations.append(
                NeuroSyntEvalDataset(
                    name=k,
                    dtype=CSVDict,
                    df=df,
                    add_stats=False,
                )
            )
        remainder = self.df[~self.df["result_tool"].isin(tools)]
        if len(remainder) != 0:
            evaluations.append(
                NeuroSyntEvalDataset(
                    name="Remainder", dtype=CSVDict, df=remainder, add_stats=False
                )
            )

        collapsed = NeuroSyntEvalDataset.from_merge(evaluations=evaluations, name="Collapsed")

        if inplace:
            self = collapsed
            return self
        else:
            return collapsed

    @classmethod
    def from_LTLSynEvalDataset(cls, dataset: LTLSynEvalDataset, **kwargs):
        df = dataset.df.rename(
            {
                "verification_mc_tool": "result_model_checking_tool",
                "prediction_circuit": "result_synthesis_circuit",
                "NeuroSynt": "result_synthesis_tool",
                "prediction_num_ands": "result_synthesis_num_ands",
                "result_synthesis_max_var_id": "result_synthesis_max_var_id",
                "result_synthesis_realizable": "result_realizable",
                "prediction_num_latches": "result_synthesis_num_latches",
                "prediction_id_AIGERCircuit": "result_synthesis_id_AIGERCircuit",
                "verification_satisfied": "result_model_checking_satisfied",
                "prediction_realizable": "result_synthesis_realizable",
                "prediction_valid": "result_valid",
                "verification_mc_time": "result_model_checking_duration",
                "syn_time_par": "result_duration_par",
                "verification_err": "result_model_checking_detailed_status",
            },
            axis=1,
        )
        df["result_tool"] = "NeuroSynt"
        return cls(dtype=CSVDict, df=df, metadata=dataset.metadata, add_stats=False, **kwargs)
