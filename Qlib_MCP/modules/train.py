import os
import json
from pathlib import Path
from typing import Dict, Any

import yaml
import joblib
import pandas as pd


def _load_config(path: str) -> Dict[str, Any]:
	path = os.path.expanduser(path)
	with open(path, "r", encoding="utf-8") as f:
		if path.endswith((".yml", ".yaml")):
			return yaml.safe_load(f)
		return json.load(f)


def _init_qlib_from_cfg(cfg: Dict[str, Any]) -> None:
	import qlib
	from qlib.config import REG_CN, REG_US

	provider_uri = os.path.expanduser(cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"))
	region = cfg.get("region", "cn")
	region_const = REG_CN if region == "cn" else REG_US
	qlib.init(provider_uri=provider_uri, region=region_const)


def _prepare_dataset_raw(cfg: Dict[str, Any]):
	raw_path = os.path.expanduser(cfg["raw_parquet"])  # Must exist
	df = pd.read_parquet(raw_path)
	# Expected MultiIndex: (datetime, instrument)
	feature_cols = cfg.get("feature_cols")
	label_col = cfg.get("label_col")
	if not feature_cols or not label_col:
		raise ValueError("raw_parquet mode requires feature_cols and label_col in configuration")
	x = df[feature_cols]
	y = df[label_col]
	return x, y


def _prepare_dataset(cfg: Dict[str, Any]):
	from qlib.contrib.data.handler import DataHandlerLP
	from qlib.contrib.data.dataset import DatasetH

	handler_cfg = {
		"start_time": cfg.get("start_time"),
		"end_time": cfg.get("end_time"),
		"instruments": cfg.get("instruments", "csi300"),
		"fields": cfg.get("features", ["$close", "$open"]),
		"filter_pipe": cfg.get("filters", []),
		"labels": [cfg.get("label", "Ref($close, -1)/$close - 1")],
	}
	dataset = DatasetH(handler=DataHandlerLP, handler_kwargs=handler_cfg)
	segments = cfg.get("segments", {"train": (cfg.get("start_time"), cfg.get("end_time"))})
	return dataset, segments


def _train_model(cfg: Dict[str, Any], model_path: str) -> None:
	from sklearn.ensemble import RandomForestRegressor

	if cfg.get("raw_parquet"):
		x_train, y_train = _prepare_dataset_raw(cfg)
		model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
		model.fit(x_train.values, y_train.values.ravel())
	else:
		dataset, segments = _prepare_dataset(cfg)
		x_train, y_train = dataset.prepare("train", col_set=["feature"], data_key=("train", segments.get("train")))
		model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
		model.fit(x_train.values, y_train.values.ravel())

	Path(os.path.dirname(model_path) or ".").mkdir(parents=True, exist_ok=True)
	joblib.dump(model, model_path)


def run(args) -> None:
	cfg = _load_config(args.config)
	if not cfg.get("raw_parquet"):
		_init_qlib_from_cfg(cfg)
	model_path = os.path.expanduser(args.model_path)
	_train_model(cfg, model_path)
	print(f"Model trained and saved: {model_path}")
