"""Build :class:`Experiment` lists from ``config/benchmark.yaml``."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from textSummarizer.evaluation.benchmark import LSTM, ORACLE, Experiment
from textSummarizer.extractive import SUMMARIZERS
from textSummarizer.preprocessing import PreprocessorConfig
from textSummarizer.service import SelectionOptions

DEFAULT_CONFIG_PATH = Path("config/benchmark.yaml")


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_experiments(
    config: dict[str, Any],
    dataset: str,
    suites: list[str],
) -> list[Experiment]:
    """Merge dataset defaults with each experiment's overrides.

    Precedence: experiment ``selection`` / ``preprocessing`` > dataset defaults > code defaults.
    """
    dataset_defaults = config.get("datasets", {}).get(dataset, {})
    default_selection = dataset_defaults.get("selection", {})
    default_preprocessing = dataset_defaults.get("preprocessing", {})

    experiments = []
    for suite in suites:
        if suite not in config.get("suites", {}):
            raise ValueError(f"Unknown suite {suite!r}; available: {sorted(config.get('suites', {}))}")
        for spec in config["suites"][suite]:
            method = spec["method"]
            if method not in (ORACLE, LSTM) and method not in SUMMARIZERS:
                raise ValueError(f"Experiment {spec.get('name')!r}: unknown method {method!r}")
            experiments.append(
                Experiment(
                    name=spec.get("name", method),
                    method=method,
                    params=spec.get("params") or {},
                    selection=_build(SelectionOptions, default_selection, spec.get("selection")),
                    preprocessing=_build(PreprocessorConfig, default_preprocessing, spec.get("preprocessing")),
                    description=spec.get("description", ""),
                )
            )

    # Ablation suites often repeat a main-suite system; keep the first.
    unique: dict[str, Experiment] = {}
    for exp in experiments:
        unique.setdefault(exp.name, exp)
    return list(unique.values())


def _build(cls, defaults: dict[str, Any], overrides: dict[str, Any] | None):
    values = {**defaults, **(overrides or {})}
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    if "extra_stopwords" in values:
        values["extra_stopwords"] = frozenset(values["extra_stopwords"])
    return cls(**values)
