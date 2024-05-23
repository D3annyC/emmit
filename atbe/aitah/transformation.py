import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import librosa
import muda
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


class AudioTransformation(object):

    def __init__(self, config_file: str, save_metadata_file_to: str = None):
        self.config = self._read_config_file(config_file)
        self.save_metadata_file_to = save_metadata_file_to
        self.deformers = []

    def _read_config_file(self, config_file: str) -> dict:
        """
        Reads and parses the configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.

        Returns:
            dict: A dictionary containing the parsed configuration data.
        """

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

    def _get_audio_paths(self, audio_format: str = ".wav", n_samples: int = 2) -> list:
        """
        Retrieves audio file paths from the MIR dataset directory.

        Args:
            audio_format (str, optional): Audio file format to search for. Defaults to ".wav".
            n_samples (int, optional): Number of audio files to retrieve. Defaults to 2.

        Raises:
            FileNotFoundError: If no audio files are found.
            ValueError: If there are insufficient audio files.

        Returns:
            list: A list of audio file paths.
        """

        dataset_path = Path(self.config["mir_dataset_path"])
        genre_paths = [d for d in dataset_path.iterdir() if d.is_dir()]
        all_sampled_paths = []

        for genre_path in genre_paths:
            audio_paths = list(genre_path.rglob(f"*{audio_format}"))
            if not audio_paths:
                raise FileNotFoundError(f"No {audio_format} files in {genre_path}")
            if len(audio_paths) < n_samples:
                raise ValueError(
                    f"Insufficient files in {genre_path}: {len(audio_paths)} available, {n_samples} requested"
                )
            sampled_paths = random.sample(audio_paths, n_samples)
            all_sampled_paths.extend(map(str, sampled_paths))

        return all_sampled_paths

    def _apply_hpss(self, audio_path: str) -> None:
        """
        Applies harmonic-percussive source separation (HPSS) to the audio file.

        Args:
            audio_path (str): Path to the audio file.
        """

        y, sr = librosa.load(audio_path, sr=None)
        y_harmonic, _ = librosa.effects.hpss(y)
        audio_directory = os.path.dirname(audio_path)
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        sf.write(os.path.join(audio_directory, f"{file_name}_hpss.wav"), y_harmonic, sr)

    def _extract_meta_from_jams(
        self, file_path: str
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extracts metadata from a JAMS file.

        Args:
            file_path (str): Path to the JAMS file.

        Raises:
            FileNotFoundError: If the JAMS file does not exist.
            ValueError: If the JAMS file is not valid JSON.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[str]]: A tuple containing the extracted metadata.
        """

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {file_path}")

        n_semitones = rate = preset = None
        transformations = data.get("sandbox", {}).get("muda", {}).get("history", [])
        for transformation in transformations:
            transformer_class = transformation.get("transformer", {}).get(
                "__class__", ""
            )
            state = transformation.get("state", {})
            if transformer_class == "LinearPitchShift":
                n_semitones = state.get("n_semitones")
            elif transformer_class == "LogspaceTimeStretch":
                rate = state.get("rate")
            elif transformer_class == "DynamicRangeCompression":
                preset = state.get("preset")

        return n_semitones, rate, preset

    def _get_meta_df(self) -> pd.DataFrame:
        """
        Retrieves metadata from the augmented audio files.

        Returns:
            pd.DataFrame: A DataFrame containing the metadata.
        """

        columns = ["file_name", "type", "n_semitones", "rate", "preset", "hpss"]
        rows = []

        file_list = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.config["augmented_meta_save_path"])
            for file in files
            if file.endswith(".jams")
        ]

        for file_name in file_list:
            n_semitones, rate, preset = self._extract_meta_from_jams(file_name)
            type_of_music = file_name.split("/")[-2]
            base_file_name = os.path.splitext(os.path.basename(file_name))[0]

            normal_row = {
                "n_semitones": n_semitones,
                "rate": rate,
                "preset": preset,
                "type": type_of_music,
                "file_name": base_file_name,
                "hpss": 0,
            }
            rows.append(normal_row)

            if self.config["hpss"]["apply"]:
                hpss_row = normal_row.copy()
                hpss_row["file_name"] = f"{base_file_name}_hpss"
                hpss_row["hpss"] = 1
                rows.append(hpss_row)

        return pd.DataFrame(rows, columns=columns)

    def _process_audio_file(self, audio_path: str, output_format: str) -> None:
        """
        Processes an audio file and applies the transformations.

        Args:
            audio_path (str): Path to the audio file.
            output_format (str): Output audio format.
        """

        j_orig = muda.load_jam_audio(jam_in=None, audio_file=audio_path)
        original_file_name = audio_path.split("/")[-1].split(".")[0]
        style = audio_path.split("/")[-2]
        audio_output_dir = os.path.join(self.config["augmented_audio_save_path"], style)
        jams_output_dir = os.path.join(self.config["augmented_meta_save_path"], style)

        if not os.path.exists(audio_output_dir):
            os.makedirs(audio_output_dir)
        if not os.path.exists(jams_output_dir):
            os.makedirs(jams_output_dir)

        pipeline = muda.Pipeline(steps=self.deformers)
        for i, jam_out in enumerate(pipeline.transform(j_orig)):
            audio_output_path = os.path.join(
                audio_output_dir, f"{original_file_name}_{i:02d}{output_format}"
            )
            jams_output_path = os.path.join(
                jams_output_dir, f"{original_file_name}_{i:02d}.jams"
            )
            muda.save(audio_output_path, jams_output_path, jam_out)

            if self.config["hpss"]["apply"]:
                self._apply_hpss(audio_output_path)

    def synthesis(
        self,
        input_format: str = ".wav",
        output_format: str = ".wav",
        n_samples: int = 2,
    ) -> None:
        """
        Synthesizes augmented audio files and extracts metadata.

        Args:
            output_format (str, optional): Output audio format. Defaults to ".wav".
            n_samples (int, optional): Number of audio files to synthesize. Defaults to 2.
        """
        self._setup_deformers()
        audio_list = self._get_audio_paths(
            audio_format=input_format, n_samples=n_samples
        )
        for audio_path in tqdm(audio_list):
            self._process_audio_file(audio_path, output_format)

        meta_df = self._get_meta_df()
        if self.save_metadata_file_to is None:
            self.save_metadata_file_to = "metadata.csv"
        meta_df.to_csv(self.save_metadata_file_to, index=False)

    def _setup_deformers(self) -> None:
        """
        Sets up the deformers based on the configuration.
        """
        if "tempo_factor" in self.config:
            self.deformers.append(
                (
                    "time_stretch",
                    muda.deformers.LogspaceTimeStretch(**self.config["tempo_factor"]),
                )
            )
        if "keys" in self.config:
            self.deformers.append(
                ("pitch_shift", muda.deformers.LinearPitchShift(**self.config["keys"]))
            )
        if "drc" in self.config:
            self.deformers.append(
                (
                    "drc",
                    muda.deformers.DynamicRangeCompression(preset=self.config["drc"]),
                )
            )
