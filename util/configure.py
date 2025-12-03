import atexit
import dataclasses
import json
import logging
import random
import sys
from logging import Formatter, Logger, StreamHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray


@dataclasses.dataclass
class Audio_Sources:
    input: Path | str
    loc: list[float]
    classification: str

    def resolve_input(self, project_dir: Path) -> None:
        self.input = project_dir / "input" / "audio" / self.input


class Config:
    # Define path to project directory
    __project_dir = Path(__file__, "../..").resolve()

    # Define string template for logger
    __log_str = f"{'Date/Time:':<19}  |  {'Level:':<8}  |  Message:\n"

    # Define log formatter
    __log_fmt = Formatter(
        fmt="{asctime:<19}  |  {levelname:<8}  |  {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    def __init__(self) -> None:
        # Parse project settings file
        with Path(self.__project_dir, "config", "config.json").open("r") as fp:
            self.__project_config = json.load(fp)

        # Define project logger
        self.__log = logging.getLogger(self.__project_dir.stem)

        # Set logger level
        self.__log.setLevel(logging.INFO)

        # Create log handler and set stream level and format
        log_hdlr = StreamHandler(sys.stdout)
        log_hdlr.setFormatter(self.__log_fmt)
        log_hdlr.setLevel(logging.INFO)

        # Add log handler to logger
        self.__log.addHandler(log_hdlr)

        # Print log string to stream
        sys.stdout.write(self.__log_str)

        # Register function to execute at end of script
        atexit.register(self.close)

        # Set all seeds
        self.set_all_seeds(self.__project_config["seed"])

        # Determine device
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print to log
        self.__log.info(f"Using device: {self.__device}")

    def set_all_seeds(self, seed: int) -> None:
        # Set seeds
        torch.manual_seed(seed)  # pyright: ignore [reportUnknownMemberType]
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        self.__rng = np.random.default_rng(seed=seed)

    @property
    def log(self) -> Logger:
        # Return logger
        return self.__log

    @property
    def rng(self) -> Generator:
        # Return number generator
        return self.__rng

    @property
    def device(self) -> torch.device:
        # Return device type
        return self.__device

    @property
    def sound_speed(self) -> float:
        # Return speed of sound
        return self.__project_config["sound_speed"]

    @property
    def input_dir(self) -> Path:
        # Return input directory path
        return Path(self.__project_dir, self.__project_config["input_dir"]).resolve()

    @property
    def output_dir(self) -> Path:
        # Return output directory path
        return Path(self.__project_dir, self.__project_config["output_dir"]).resolve()

    @property
    def checkpoint_file(self) -> Path | None:
        # Check if checkpoint file is defined
        if self.__project_config["checkpoint_file"]:
            checkpoint_file_path = Path(self.__project_config["checkpoint_file"]).resolve()
        else:
            checkpoint_file_path = None

        # Return checkpoint file path
        return checkpoint_file_path

    @property
    def train_max_sessions(self) -> int | None:
        # Set max sessions to none if set to zero
        max_sessions = self.__project_config["training"]["max_sessions"]
        if max_sessions == 0:
            max_sessions = None

        # Return max sessions for training
        return max_sessions

    @property
    def val_max_sessions(self) -> int | None:
        # Set max sessions to none if set to zero
        max_sessions = self.__project_config["validation"]["max_sessions"]
        if max_sessions == 0:
            max_sessions = None

        # Return max sessions for validation
        return max_sessions

    @property
    def train_batch_size(self) -> int:
        # Return batch size for training
        return self.__project_config["training"]["batch_size"]

    @property
    def val_batch_size(self) -> int:
        # Return batch size for validation
        return self.__project_config["validation"]["batch_size"]

    @property
    def optimizer_params(self) -> dict[str, Any]:
        # Return optimizer parameters
        return self.__project_config["optimizer"]

    @property
    def scheduler_params(self) -> dict[str, Any]:
        # Return scheduler parameters
        return self.__project_config["scheduler"]

    @property
    def epoch_count(self) -> int:
        # Return epoch count
        return self.__project_config["epoch_count"]

    @property
    def fs(self) -> int:
        # Return sampling frequency
        return self.__project_config["sampling_frequency"]

    @property
    def room_dim(self) -> NDArray[np.float64]:
        # Return room dimensions
        return np.array(
            list(self.__project_config["room_settings"]["dim"].values()), dtype=np.float64
        )

    @property
    def reflection_count(self) -> int:
        # Return room reflection count
        return self.__project_config["room_settings"]["reflection_count"]

    @property
    def mic_count(self) -> int:
        # Return number of microphones
        return self.__project_config["mic_settings"]["count"]

    @property
    def mic_spacing(self) -> float:
        # Return spacing between microphones
        return self.__project_config["mic_settings"]["spacing"]

    @property
    def mic_loc(self) -> NDArray[np.float64]:
        # Return location of microphones
        return np.array(self.__project_config["mic_settings"]["loc"], dtype=np.float64)

    @property
    def mic_type(self) -> str:
        # Return type of microphones
        return self.__project_config["mic_settings"]["type"]

    @property
    def sources(self) -> list[Audio_Sources]:
        # Concatenate list of sources
        sources = [Audio_Sources(**source) for source in self.__project_config["sources"]]
        for source in sources:
            source.resolve_input(self.__project_dir)

        # Return list of sources
        return sources

    @property
    def noise_pc_count(self) -> int:
        # Return number of principal components to reduce noise
        return self.__project_config["noise_pc_count"]

    @property
    def noise_reg_factor(self) -> float:
        # Return noise regularization factor
        return self.__project_config["noise_reg_factor"]

    @property
    def frame_duration(self) -> float:
        # Return audio frame duration in milliseconds
        return self.__project_config["frame_duration"]

    def close(self) -> None:
        # delete package logger and close package log
        for log_hdlr in self.__log.handlers:
            self.__log.removeHandler(log_hdlr)
            log_hdlr.close()

        # Delete log
        del self.__log
