# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:34:48 2023

@author: Administrator
"""

import whisperx
import gc
import os.path
import soundfile as sf
import numpy as np
import samplerate
from scipy.signal import butter, lfilter
from typing import Optional, Union, List
from pydub import AudioSegment
import ffmpeg
import argparse
import time

from Simplify_Chinese.convert_simple import convert_t2s

class WisperAPI:   
    def __init__(self, model_path: str, device: str="cuda", language: str='zh'):
        self.model = whisperx.load_model(model_path, device=device, compute_type="float32", language=language)
    
    # 高斯滤波，提高准确度(可选)
    def highpass_filter(self, data: np.array, sr: int, 
                        cutoff: int=100, order: int=5) -> np.array:
        # 设计高通滤波器
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
        # 对数据应用滤波器
        filtered_data = lfilter(b, a, data)
        return filtered_data
    
    def read_mp3(self, filepath: str, dtype: str='float32') -> np.array:
        # 使用pydub读取.mp3文件
        audio = AudioSegment.from_mp3(filepath)

        # 将音频转换为numpy数组
        samples = np.array(audio.get_array_of_samples(), dtype=dtype)
        if audio.sample_width == 2:
            samples = samples / (2**15)
        elif audio.sample_width == 1:
            samples = (samples - 128) / (2**7)  # 8位音频的处理方式略有不同
        
        return samples, audio.frame_rate * 2
    
    # 带有冗余项的音频切割
    def split_audio_with_overlap(self, data: np.array, file_sr: int, 
                                 segment_duration: int=10, overlap_duration: int=2) -> List[str]:
        # Calculate the number of samples per segment without the overlap
        non_overlap_samples = int((segment_duration - overlap_duration) * file_sr)
        # Calculate the total number of samples per segment including the overlap
        samples_per_segment = int(segment_duration * file_sr)
        # Calculate the number of segments
        number_of_segments = int(np.ceil(len(data) / non_overlap_samples))
    
        # Create a list to hold the segments
        segments = []
    
        for i in range(number_of_segments):
            # Calculate the start and end of the current segment
            start = i * non_overlap_samples
            end = start + samples_per_segment
            # Make sure we don't go past the end of the array
            end = min(end, len(data))
            # Append the current segment to the list
            segments.append(data[start:end])
    
        return segments
    
    def audio_to_numpy(self, filepath: str,
                       dtype: str='float32') -> [np.array, int]:
        
        if '.wav' in filepath: 
            data, file_sr = sf.read(filepath, dtype='float32')
        elif '.mp3' in filepath:
            data, file_sr = self.read_mp3(filepath, dtype='float32')
        return data, file_sr
    
    # 读取音频，返回np.ndarray列表
    def audio_to_split_list(self, filepath: str,
                       sr: int = 16000) -> list:
        
        data, file_sr = self.audio_to_numpy(filepath, dtype='float32')

        # 可选，高通滤波
        data = self.highpass_filter(data, file_sr)
    
        # 确保音频是单声道
        if len(data.shape) == 2:
            data = data.mean(axis=1)
    
    
        # 如果音频文件的采样率与目标采样率不符，进行重新采样
        if file_sr != sr:
            data = samplerate.resample(data, sr / file_sr, 'sinc_best')

    
        # 折叠data，每段60s
        data = self.split_audio_with_overlap(data, file_sr)
    
        for idx, data_int16 in enumerate(data):
            # if file_sr != sr:
            #     data_int16 = samplerate.resample(data_int16, sr / file_sr, 'sinc_best')
            # 转换为int16并归一化
            data_int16 = (data_int16 * 32768.0).astype(np.int16)
            data[idx] = data_int16.astype(np.float32) / 32768.0
        return data
    
    def match_and_combine(self, sentences: list, match_left_corner: int=20, 
                          min_match_length: int=2) -> str:
        """
        Combine sentences by matching and removing the overlapping parts.
        :param sentences: List of sentences to combine
        :param min_match_length: Minimum length of match to consider for combining
        :return: List of combined sentences
        """
        n = len(sentences)
    
        for i in range(n):
            # If it's the last sentence, just add it to the list
            if i == n-1:
                break
            
            # Find the longest common substring that is at the end of one sentence and the start of the next
            s1, s2 = sentences[i], sentences[i+1]
            match = self.longest_common_substring(s1[len(s1)-int(len(s1)*0.4):], 
                                                             s2[:int(len(s2)*0.3)] if len(s2) >= 20 else s2)
    
            # Check if the match is long enough
            if len(match) >= min_match_length:
                # Remove the match from the end of the first sentence and the start of the second
                sentences[i] = s1[:s1.rfind(match)]
                sentences[i+1] = s2[s2.find(match):]
            else:
                # If there's no match, just add the first sentence and proceed
                sentences[i] = sentences[i][:len(sentences[i])-1]
                
        combined_sentences = "".join(sentences)
                
        return combined_sentences
    
    def longest_common_substring(self, s1: str, s2: str) ->str:
        """
        Find the longest common substring between two strings using dynamic programming.
        :param s1: First string
        :param s2: Second string
        :return: The longest common substring
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        longest, x_longest = 0, 0
    
        for x in range(1, m+1):
            for y in range(1, n+1):
                if s1[x-1] == s2[y-1]:
                    dp[x][y] = dp[x-1][y-1] + 1
                    if dp[x][y] > longest:
                        longest = dp[x][y]
                        x_longest = x
                else:
                    dp[x][y] = 0
    
        return s1[x_longest-longest: x_longest]
    

    def run(self, audio: np.array, 
            batch_size: int=16,
            language: str='zh') -> str:
        
        segment_text = ""
        start_time = time.time()
        transcription = self.model.transcribe(audio,
                                          batch_size=16, 
                                          language='zh'
                                          )

        transcription = list(transcription["segments"])
        for aud in transcription:
            segment_text += aud["text"]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n执行时间为：{elapsed_time}秒:")
        transcriptions = convert_t2s(segment_text)
        return transcriptions
        
        # transcriptions = self.match_and_combine(transcriptions)
        # return transcriptions

if __name__ != "__main__":
    pass
else:
    parser = argparse.ArgumentParser(description="whisper")
    parser.add_argument('--language', dest='language', type=str, default='Chinese')
    parser.add_argument('--model_type', dest='model_type', type=str, default='large')
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    parser.add_argument('--model_path', dest='model_path', type=str, default='E:/code/faster-whisper/faster-whisper-large-v2')
    parser.add_argument('--audio_path', dest='audio_path', type=str, default='E:/code/whisper/test4.wav')
    args = parser.parse_args()
    
    # load whisper model
    whisper_api = WisperAPI(args.model_path)
    
    audio = whisper_api.audio_to_split_list(args.audio_path)
    
    for a in audio:
        transcriptions = whisper_api.run(a)
        print(transcriptions)
      
    # print(f"最终的段落: {transcriptions}")
    
    
    
    
    
    
    
    
    
    
    