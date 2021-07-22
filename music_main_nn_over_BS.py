# from visualize import save_matrices, save_dance
from multiprocessing import Pool
import sys
from PIL import Image
import numpy as np
import itertools
import argparse
import random
import scipy
import os
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import wave
import simpleaudio as sa
import torch
import torch.nn as nn

join = os.path.join

random.seed(123)
num_classes = 5


def get_mapped_note(note_idx):
    mappings = {0: 60, 1: 62, 2: 64, 3: 67, 4: 69}
    return mappings[note_idx]

def get_note_wave(note_val, duration=10.0):
    wave_read = wave.open("music_files/" + str(note_val) + ".wav", 'rb')
    wave_obj = sa.WaveObject.from_wave_read(wave_read)

    return wave_obj


def take_action(cur_note, prev_note, prev_play_obj, note_waves):
    if prev_note == -1 or cur_note != prev_note:
        print("note changed")
        if prev_note != -1:
            prev_play_obj.stop()
        play_obj = note_waves[cur_note].play()
        return play_obj, 1
    return prev_play_obj, 0

def preprocess_dance_matrix(dance_matrix):
    l = len(dance_matrix)
    if l != 60:
        temp = np.zeros([60, 60])
        temp[-l:,-l:] = dance_matrix
        return np.expand_dims(temp, axis=0)
    else:
        return np.expand_dims(dance_matrix, axis=0)

def preprocess_past_music(past_music):
    temp = [num_classes]*10 + list(past_music)
    temp = temp[-10:]
    new_past_music = np.array(temp)
    return np.expand_dims(new_past_music, axis=0)

class Net(nn.Module):
    def __init__(self):
        # call init function of parent class
        super(Net, self).__init__()
        # define / initialize layers of network
        self.embedding_dim = embedding_dim = 16
        self.latent_dim_dance = latent_dim_dance = 32
        self.latent_dim_music = latent_dim_music = 32
        self.use_batch_norm = use_batch_norm = False
        self.use_lstm = use_lstm = True

        if use_batch_norm:
            self.danceNet = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 3),
        nn.ReLU(True),
        nn.Conv2d(512, latent_dim_dance,  3)
            )
        else:
            self.danceNet = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 3),
                nn.ReLU(True),
                nn.Conv2d(512, latent_dim_dance,  3)
            )

        if use_lstm:
            self.musicNet = nn.LSTM(self.embedding_dim, latent_dim_music, 2,bidirectional=True)
        else:
            self.musicNet = nn.Sequential(
                nn.Linear(num_classes * self.embedding_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, latent_dim_music)
            )

        self.danceMusicNet = nn.Sequential(
            nn.Linear(latent_dim_dance + 2*latent_dim_music, 512),
            # nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
        self.notes_embedding = nn.Embedding(6, self.embedding_dim)

    # defines the computation performed at every call of the module
    # this is not called directly, but via instance of Module
    def forward(self, d, m):
        dance_out = d
        music_out = m
        # how to pass input here
        dance_out = self.danceNet(dance_out)
        dance_out = torch.mean(dance_out, (2, 3))
        # 32*512
        # return dance_out
        music_out = self.notes_embedding(music_out)
        if self.use_lstm:
            batch_size = d.size(0)
            h0 = torch.zeros(4, batch_size, self.latent_dim_music)
            c0 = torch.zeros(4, batch_size, self.latent_dim_music)
            music_out = music_out.permute(1, 0, 2)
            music_out, _ = self.musicNet(music_out, (h0, c0))
            music_out = music_out[-1]
        else:
            music_out = music_out.reshape((music_out.shape[0], -1))
            music_out = self.musicNet(music_out)
        dance_music_concat = torch.cat((dance_out, music_out), 1)
        dance_music_out = self.danceMusicNet(dance_music_concat)
        return dance_music_out

class Music(object):
    def __init__(self, nn_frame_start, music_freq, joint_idx=None, get_note=False):
        self.nn_frame_start = nn_frame_start
        self.music_freq = music_freq
        self.joint_idx = joint_idx

        note_mappings = {60: 'C4', 62: 'D4', 64: 'E4', 67: 'G4', 69: 'A4'}

        note_waves = {}

        for key in note_mappings:
            note_waves[key] = get_note_wave(key)

        self.note_waves = note_waves
        self.prev_note = -1
        self.prev_play_obj = None
        self.net = Net()
        print("loading the model")
        self.net.load_state_dict(torch.load("models/models_32_32_16_500_0.0002", map_location=torch.device('cpu'))["state_dict"])
        self.net.eval()
        print("the model has been loaded")
        self.joints = []
        self.splits = []
        self.get_note = get_note
        self.past_music = []
        self.music_history = 10
        self.joints_history = 60

        

    def add_pose(self, pose, frame_idx):

        self.joints.append(pose)
        print("the frame idx is ", frame_idx)

        if frame_idx % self.music_freq == 0:
            if len(self.past_music) == 0:
                note_idx = 2
                
            else:
                if len(self.joints) >= self.joints_history:
                    self.joints = self.joints[-self.joints_history:]
                joints_similarity = cosine_similarity(np.array(self.joints))

                if len(self.past_music) < 10:
                    joint_matrix = joints_similarity
                    past_music = self.past_music
                else:
                    joint_matrix = joints_similarity
                    past_music = self.past_music[-10:]

                print("the shape of joint_matrix is ", joint_matrix.shape)
                joint_matrix, past_music = preprocess_dance_matrix(joint_matrix), preprocess_past_music(past_music)
                joint_matrix = torch.from_numpy(np.expand_dims(joint_matrix, 1)).float()
                past_music = torch.from_numpy(past_music).long()
                out = self.net(joint_matrix, past_music)
                nn_note = out.cpu().data.numpy().argmax(axis=1)[0]

                note_idx = nn_note

            print("the music produced is ", note_idx)
            state = get_mapped_note(note_idx)
            self.past_music.append(note_idx)
            cur_note = state
            self.prev_play_obj, note_change = take_action(cur_note, self.prev_note, self.prev_play_obj, self.note_waves)
            self.prev_note = cur_note
            return note_change

        return 0


if __name__ == "__main__":

    music_obj = Music(50, 6, 9)
