

import os
import torch
import torch.nn as nn
import torchaudio
import yaml
import requests 
import warnings
from transformers import VitsTokenizer, VitsModel, set_seed


from typing import Optional
from typing import Union
from typing import List

import numpy as np

from tensortract2 import TensorTract2
from vocaltractlab import motor_to_audio, motor_to_contour
from target_approximation.vocaltractlab import MotorSeries as VTL_MSRS
from target_approximation.tensortract import MotorSeries as TT_MSRS


    
class TensorTractLab( TensorTract2 ):
    def __init__(
            self,
            cfg_path: str = 'tensortract2_version_uc81_am100',
            auto_load_weights: bool = True,
            hf_token: Optional[str] = None,
            ):
        super().__init__(
            cfg_path=cfg_path,
            auto_load_weights=auto_load_weights,
            hf_token=hf_token,
            )
        self.tts_tokenizer_en = VitsTokenizer.from_pretrained(
            "facebook/mms-tts-eng",
            token = hf_token,
            )
        self.tts_model_en = VitsModel.from_pretrained(
            "facebook/mms-tts-eng",
            token = hf_token,
            )
        return
    
    def motor_to_speech(
            self,
            msrs,
            target,
            output: Optional[ Union[ str, List[str] ] ] = None,
            output_vtl: Optional[ Union[ str, List[str] ] ] = None,
            time_stretch: Optional[ float ] = None,
            pitch_shift: Optional[ float ] = None,
            #msrs_type: str = 'tt2', #must be vtl here
            synthesis_type: str = 'tt2',
            vtl_kwargs: dict = {
                'normalize_audio': -1,
                'sr': 16000,
                'return_data': True,
                'workers': None,
                'verbose': True,
                },
            ):
        """
        Convert motor series objects to speech.

        Args:
            msrs: List[
                Union[
                    target_approximation.vocaltractlab.MotorSeries,
                    target_approximation.tensortract.MotorSeries,
                    ],
                ], List of motor series objects.
            target: Union[str,List[str]], Path to audio file or list of paths.
            output: Union[str,List[str]], Path to output file or list of paths.
            time_stretch: float, Latent time-stretching factor. E.g. to double
                the speed, set to 2. Default is None, which means no time-stretching
                will be applied.
            pitch_shift: float, Latent pitch shift value in semitones. E.g. to shift
                pitch up/down an octave, set to +/- 12. Default is None, which
                means no pitch shift will be applied.
            msrs_type: str, Expected type of motor series input. Must be 'tt2' or 'vtl'.
                'vtl' means a VTL-Python standard (30 articulatory features at 441 Hz).
                'tt2' means a TensorTract2 standard (20 articulatory features at 50 Hz).

        Returns:
            x: torch.Tensor, (B,T). Audio waveform tensor, 16kHz.
            x_len: torch.Tensor, (B,). Length of each output tensor.
        """
        if isinstance(msrs, TT_MSRS):
            raise ValueError(
                f"""
                Expected VTL motor series, but received TT2 motor series.
                """
                )
        # Manipulate the motor series objects here,
        # so that they can be used by both VTL and TT2.
        for ms in msrs:
            if time_stretch is not None:
                ms.time_stretch( time_stretch )
            if pitch_shift is not None:
                ms.pitch_shift( pitch_shift )
        # TT2 synthesis
        if synthesis_type in ['tt2', 'both']:
            x = super().motor_to_speech(
                msrs,
                target,
                output=output,
                time_stretch=None,
                pitch_shift=None,
                msrs_type='vtl',
                )
        else:
            x = None
        # VTL synthesis
        if synthesis_type in ['vtl', 'both']:
            y = motor_to_audio(
                motor_data = msrs,
                audio_files = output_vtl,
                **vtl_kwargs,
                )
        else:
            y = None
        return x, y
    
    def speech_to_speech(
            self,
            x: Union[
                str,
                List[str],
                ],
            target: Optional[ Union[ str, List[str] ] ] = None,
            output: Optional[ Union[ str, List[str] ] ] = None,
            output_vtl: Optional[ Union[ str, List[str] ] ] = None,
            time_stretch: Optional[ float ] = None,
            pitch_shift: Optional[ float ] = None,
            synthesis_type: str = 'tt2',
            vtl_kwargs: dict = {
                'normalize_audio': -1,
                'sr': 16000,
                'return_data': True,
                'workers': None,
                'verbose': True,
                },
            export_video: str = None,
            audio_track: str = None,
            ):
        """
        Convert speech to speech, i.e. re-synthesis or voice conversion.

        Args:
            x: Union[str,List[str]], Path to audio file or list of paths.
            target: Union[str,List[str]], Path to audio file or list of paths.
                Default is None, which means the target speaker voice is the
                same as in x.
            output: Union[str,List[str]], Path to output file or list of paths.
                Default is None, which means no output will be saved.
            time_stretch: float, Latent time-stretching factor. E.g. to double
                the speed, set to 2. Default is None, which means no time-stretching
                will be applied.
            pitch_shift: float, Latent pitch shift value in semitones. E.g. to shift
                pitch up/down an octave, set to +/- 12. Default is None, which
                means no pitch shift will be applied.
            
        Returns:
            x: torch.Tensor, (B,T). Audio waveform tensor, 16kHz.
            x_len: torch.Tensor, (B,). Length of each output tensor.
        """
        msrs = super().speech_to_motor(x, msrs_type='vtl')
        if target is None:
            target = x

        if output is not None:
            output_dirname = os.path.dirname(output)
            if output_dirname != '':
                os.makedirs(output_dirname, exist_ok=True)
        if output_vtl is not None:
            output_vtl_dirname = os.path.dirname(output_vtl)
            if output_vtl_dirname != '':
                os.makedirs(output_vtl_dirname, exist_ok=True)
        if export_video is not None:
            export_video_dirname = os.path.dirname(export_video)
            if export_video_dirname != '':
                os.makedirs(export_video_dirname, exist_ok=True)

        x, y = self.motor_to_speech(
            msrs,
            target,
            output=output,
            output_vtl=output_vtl,
            time_stretch=time_stretch,
            pitch_shift=pitch_shift,
            synthesis_type=synthesis_type,
            vtl_kwargs=vtl_kwargs,
            )
        if export_video is not None:
            motor_to_contour(
                msrs[0],
                video_file = export_video,
                audio_file = audio_track,
                fps=30,
                synthesize=False,
                )
        return x, y
    
    def text_to_motor(
            self,
            text: str,
            output_text: str,
            msrs_type: str = 'tt2',
            ):
        """
        Convert text to motor series.

        Args:
            text: str, Text to convert.
            msrs_type: str, Expected type of motor series output. Must be 'tt2' or 'vtl'.
                'vtl' means a VTL-Python standard (30 articulatory features at 441 Hz).
                'tt2' means a TensorTract2 standard (20 articulatory features at 50 Hz).

        Returns:
            x: torch.Tensor, (B,T). Audio waveform tensor, 16kHz.
            x_len: torch.Tensor, (B,). Length of each output tensor.
            msrs: target_approximation.vocaltractlab.MotorSeries or
                target_approximation.tensortract.MotorSeries, Motor series object.
        """
        # Get the audio from the text using TTS
        inputs = self.tts_tokenizer_en(text=text, return_tensors="pt")
        set_seed(555)
        with torch.no_grad():
            outputs = self.tts_model_en(**inputs)

        x = outputs.waveform[0]
        x = x.squeeze()
        # Save the audio to a file
        # if dirname not empty, make the directory
        dirname = os.path.dirname(output_text)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

        torchaudio.save(
            output_text,
            x.unsqueeze(0),
            self.tts_model_en.config.sampling_rate,
            )

        msrs = super().speech_to_motor(
            output_text,
            msrs_type=msrs_type,
            )
        return msrs
    
    def text_to_speech(
            self,
            text: str,
            output_text: str,
            target: Optional[ Union[ str, List[str] ] ] = None,
            output: Optional[ Union[ str, List[str] ] ] = None,
            output_vtl: Optional[ Union[ str, List[str] ] ] = None,
            time_stretch: Optional[ float ] = None,
            pitch_shift: Optional[ float ] = None,
            synthesis_type: str = 'tt2',
            vtl_kwargs: dict = {
                'normalize_audio': -1,
                'sr': 16000,
                'return_data': True,
                'workers': None,
                'verbose': True,
                },
            export_video: str = None,
            audio_track: str = None,
            ):
        msrs = self.text_to_motor(
            text,
            output_text,
            msrs_type='vtl',
            )
        if target is None:
            target = output_text

        if output is not None:
            output_dirname = os.path.dirname(output)
            if output_dirname != '':
                os.makedirs(output_dirname, exist_ok=True)
        if output_vtl is not None:
            output_vtl_dirname = os.path.dirname(output_vtl)
            if output_vtl_dirname != '':
                os.makedirs(output_vtl_dirname, exist_ok=True)
        if export_video is not None:
            export_video_dirname = os.path.dirname(export_video)
            if export_video_dirname != '':
                os.makedirs(export_video_dirname, exist_ok=True)
                
        x, y = self.motor_to_speech(
            msrs,
            target,
            output=output,
            output_vtl=output_vtl,
            time_stretch=time_stretch,
            pitch_shift=pitch_shift,
            synthesis_type=synthesis_type,
            vtl_kwargs=vtl_kwargs,
            )
        if export_video is not None:
            motor_to_contour(
                msrs[0],
                video_file = export_video,
                audio_file = audio_track,
                fps=30,
                synthesize=False,
                )
        return x, y