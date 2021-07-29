"""Script to generate all dances for a song."""
# from visualize import save_matrices, save_dance
from multiprocessing import Pool
from PIL import Image
import numpy as np
import itertools
import argparse
import librosa
import random
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import note_seq
import os
from visualize_music import save_video, generate_mp3, save_matrices, save_music
from note_seq.protobuf import music_pb2

from time import time

join=os.path.join

random.seed(123)

# parse arguments
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-dancepath', '--dancepath', type=str, default='./feats_2/ballet/',
                    help='Path to .npy dance')
parser.add_argument('-imagespath', '--imagespath', type=str, default='./dance_images_1/ballet',
                    help='Path to .npy dance')
parser.add_argument('-featspath', '--featspath', type=str, default='./feats/polyphony_feats_pentatonic.npy',
                    help='Path to music features')

parser.add_argument('-history', '--history', type=int, default=10,
                    help='Number of equidistant LRS steps the agent should take')

parser.add_argument('-fps', '--fps', type=int, default=30,
                    help='fps of the dance')

parser.add_argument('-type', '--type', type=str, default='state_only',
                    help='Type of dance -- state, state_embedding, action, stateplusaction, random')
# parser.add_argument('--total_frames', type=int, default=300, help='total number of dance frames to save as images')
parser.add_argument('--music_freq', type=int, default=6, help='frequency with which to play a note')

parser.add_argument('--beam_window', type=int, default=50, help='beam search window')

parser.add_argument('-baseline', '--baseline', type=str, default='none',
                    help='Generate baseline -- none, unsync_random, unsync_sequential, sync_sequential, sync_random')
parser.add_argument('-visfolder', '--visfolder', type=str, default='./results',
                    help='path to folder containing agent visualizations')
parser.add_argument('--no_openvino', action='store_true', default=False)
parser.add_argument('--chords', action='store_true', default=False)
parser.add_argument('--max', action='store_true', default=False)
parser.add_argument('--no_baseline', action='store_true', default=True)
parser.add_argument('--show_matrices', action='store_true', default=False)
parser.add_argument('--realtime', action='store_true', default=False)


args = parser.parse_args()

# global variables
if args.chords:
    GRID_SIZE = 6
else:
    GRID_SIZE = 5

REWARD_INTERVAL = 1
# ALL_ACTION_COMBS = out = list(set(itertools.permutations([0]*5+[1]*5+[2]*5+[3]*5+[4]*5, REWARD_INTERVAL)))
START_POSITION = 2
history = args.history
beam_window = args.beam_window
if args.realtime:
    beam_window = 1

# **************************************************************************************************************** #
# DANCE MATRIX CREATION


def fill_music_aff_matrix_diststate(states):
    """Fill state action affinity matrix - relative distance based states."""
    s = len(states)
    rowtile = np.tile(states, (s, 1))
    coltile = rowtile.T
    sa_aff = 1. - np.abs(rowtile-coltile) / (GRID_SIZE-1)
    return sa_aff

def fill_music_aff_matrix_state_only(states):
    return fill_music_aff_matrix_diststate(states)


def get_music_matrix(states, actions, music_matrix_type, dance_matrix_full):
    """Pass to appropriate dance matrix generation function based on music_matrix_type."""
    
    if music_matrix_type == 'state_only':
        music_matrix = fill_music_aff_matrix_state_only(states)
    else:
        print("err")
    music_matrix = np.array(Image.fromarray(np.uint8(music_matrix * 255)).resize(dance_matrix_full.shape, Image.NEAREST)) / 255.
    return music_matrix

# **************************************************************************************************************** #
# MUSIC MATRIX COMPUTATION


def compute_dance_matrix(fname):
    data = np.load(fname)
    if not args.no_openvino:
        data = np.concatenate((data[:,:16], data[:,18:38]), axis=-1)
    print(data.shape)
    similarity = cosine_similarity(data)
    print(similarity.shape)
    return similarity, data

def compute_music_matrix():
    keys = [60, 62, 64, 67, 69]
    print(keys)
    state_dict = {}
    for idx, key in enumerate(keys):
        state_dict[idx] = key    
    return state_dict


# **************************************************************************************************************** #
# REWARD COMPUTATION


def music_reward(music_matrix, dance_matrix, mtype):
    """Return the reward given music matrix and dance matrix."""
    # compute distance based on mtype
    if mtype == 'pearson':
        if np.array(music_matrix).std() == 0 or np.array(dance_matrix).std() == 0:
            reward = 0
        else:
            reward, p_val = scipy.stats.pearsonr(music_matrix.flatten(), dance_matrix.flatten())
    elif mtype == 'spearman':
        reward, p_val = scipy.stats.spearmanr(music_matrix.flatten(), dance_matrix.flatten())
    else:
        print("err")
    return reward


def get_reward(dance_matrix, states):
    if len(states) > history:
        new_states = states[-history:]
    else:
        new_states = states

    music_matrix = fill_music_aff_matrix_state_only(new_states)
    music_matrix = np.array(Image.fromarray(np.uint8(music_matrix * 255)).resize(dance_matrix.shape, Image.NEAREST)) / 255.
    # check how good dance up till now is by computing reward
    curr_reward = music_reward(dance_matrix, music_matrix, 'pearson')
    return curr_reward


def get_reward_entire(dance_matrix, states):

    music_matrix = fill_music_aff_matrix_state_only(states)
    music_matrix = np.array(Image.fromarray(np.uint8(music_matrix * 255)).resize(dance_matrix.shape, Image.NEAREST)) / 255.
    # check how good dance up till now is by computing reward
    curr_reward = music_reward(dance_matrix, music_matrix, 'pearson')
    return curr_reward

def getbest(loc, num_actions, prev_states, prev_actions, dance_matrix_full, num_steps, music_matrix_type):
    """Return best combination of size num_actions.

    Start from `loc` in grid of size `GRID_SIZE`.
    """
    s = time()
    if len(prev_states) == 0:
        return [[2]], [[2]], 0

    scale = int(dance_matrix_full.shape[0] * (np.array(prev_states).shape[1]+num_actions) / num_steps)
    scale_history = int(dance_matrix_full.shape[0] * (history) / num_steps)
    # print("the scale is ", scale, " len prev states is ", len(prev_states), " num actions is ", num_actions, "num steps is ", num_steps)
    # print("the shape is ", dance_matrix_full.shape)
    

    dance_matrix = np.array([dance_matrix_full[i][:scale] for i in range(scale)])
    if scale > scale_history:
        dance_matrix = dance_matrix[-scale_history:,-scale_history:]

    print("the scale is ", dance_matrix.shape[-1])

    # print("the shape of dance matrix is ", dance_matrix.shape)
    # get best dance for this music matrix
    bestreward = 0
    new_states = []
    rewards = []
    
    # print("the shape of prev states is ", np.array(prev_states).shape)
    for prev_state in prev_states:
        for i in range(GRID_SIZE):

            new_state = list(prev_state) + [i]
            new_states.append(new_state)
            rewards.append(get_reward(dance_matrix, new_state))

    new_states = np.array(new_states)
    indices = np.argsort(rewards)[::-1]
    new_states = new_states[indices[:beam_window]]

    print('time taken is ', time() - s)
    return new_states, new_states, rewards[indices[0]]

# **************************************************************************************************************** #
# MAIN

def process(states):
    new_states = []
    i = 0
    while i < len(states):
        new_states.append(states[i])
        j = i + 1
        while j < len(states) and states[j] == states[i]:
            new_states.append(-2)
            j += 1
        i = j

    return new_states

instrument = 0
velocity = 100

def add_note(music, pitch, start_time, end_time):
    music.notes.add(pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity, instrument=instrument)

def add_chord(music, pitches, start_time, end_time):
    for pitch in pitches:
        music.notes.add(pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity, instrument=instrument)

def process_notes_chords(states, qpm, start_time=0.0):
    duration = 15.0/float(qpm)
    print("the duration is ", duration)
    music = music_pb2.NoteSequence()
    i = 0
    # start_time = start_time
    while i < len(states):
        j = i + 1
        c = 1
        while j < len(states) and states[j] == states[i]:
            c += 1
            j += 1
        end_time = start_time + c * duration
        if states[i] == 100:
            print("playing chord and the repeat is ", c)
            add_chord(music, [60, 64, 67], start_time, end_time)
        else:
            print(states[i], start_time, end_time)
            add_note(music, states[i], start_time, end_time)
        start_time = end_time

        i = j
    music.total_time = end_time
    music.tempos.add(qpm=qpm)
    print("the end time is ", end_time)
    return music

if __name__ == "__main__":

    # get args
    
    baseline = args.baseline
    music_matrix_type = args.type
    visfolder = args.visfolder
    fps = args.fps

    if not os.path.exists(visfolder):
        os.makedirs(visfolder)

    # qpm = (15.0 * num_steps)/(args.total_frames/float(fps))
    qpm = 15.0 *float(fps)/float(args.music_freq)
    print("the qpm is ", qpm)

    state_dict = {}

    if args.chords:
        keys = [100, 60, 62, 64, 67, 69]
    else:
        keys = [60, 62, 64, 67, 69]
    for idx, key in enumerate(keys):
        state_dict[idx] = key   
    correlations = [] 

    for fname in os.listdir(args.imagespath):
        if ".ds" in fname.lower():
            continue
        output_dir = visfolder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("the output dir is ", output_dir)
        dancepath = join(args.dancepath, fname + ".npy")
        print("the dancepath is ", dancepath)

        
        dance_matrix_full, joints_data = compute_dance_matrix(dancepath)
        
        print("the shape of joints data is ", joints_data.shape)

        prev_states = []
        prev_actions = []
        music_freq = args.music_freq
        num_steps = int(len(dance_matrix_full)/music_freq)
        print("the num steps is ", num_steps)

        for i in range(num_steps):
            # apply greedy algo to get dance matrix with best reward
            print(i)
            
            prev_states, prev_actions, reward = getbest(loc=START_POSITION,
                                                        num_actions=REWARD_INTERVAL,
                                                        prev_states=prev_states,
                                                        prev_actions=prev_actions,
                                                        num_steps=num_steps,
                                                        music_matrix_type=music_matrix_type,
                                                        dance_matrix_full=dance_matrix_full)

        # get best music matrix
        music_matrix = get_music_matrix(prev_states[0], prev_actions, music_matrix_type, dance_matrix_full)

        print(dance_matrix_full.shape)
        print(music_matrix.shape)

        print("the reward is ", reward)
        


        # # assign states and actions correctly correctly
        states = prev_states[0]
        correlations.append(get_reward_entire(dance_matrix_full, states))
        baseline_indices = []
        if not args.no_baseline:
            if args.max:
                for i in range(1, len(states)):
                    idx = (i+1)*music_freq - 1
                    sim = np.max(np.abs(joints_data[idx] - joints_data[idx - music_freq]))
                    if sim < 0.087145:
                        states[i] = states[i-1]
                        baseline_indices.append(i)
            else:
                for i in range(1, len(states)):
                    idx = (i+1)*music_freq - 1
                    if dance_matrix_full[idx, idx - music_freq] >= 0.99561:
                        states[i] = states[i-1]
                        baseline_indices.append(i)

        else:
            print("not doing any post processing")
            
        states = [state_dict[s] for s in states]

        print("the states are ", states)
        print("the processed states are ", process(states))
        primer_sequence = process_notes_chords(states, qpm, start_time=6.0/30.0)


        end_time = float(primer_sequence.notes[-1].end_time)
        print("the end time is ", end_time)
        print("the instrument is ", primer_sequence.notes[-1].instrument)
        print("the velocity is ", primer_sequence.notes[-1].velocity)

        midi_fname = join(output_dir, fname + '.mid')

        print("the midi fname is ", midi_fname)
        note_seq.sequence_proto_to_midi_file(primer_sequence, 
            midi_fname)

        dance_images_dir = join(args.imagespath, fname)
        out_fname = generate_mp3(midi_fname)
        save_video(out_fname, end_time, len(dance_matrix_full), dance_images_dir, out_fname.replace(".mp3", ".mp4"))
        #save_music(out_fname.replace(".mp3", ".html"), fps, states, baseline_indices)

        if args.show_matrices:
            out_matrix_fname = out_fname.replace(".mp3", "_")
            end_time = int(len(dance_matrix_full)/fps)
            save_matrices(dance_matrix_full, music_matrix, end_time, out_matrix_fname)

        os.remove(midi_fname)
        os.remove(out_fname)
    print("the mean correlation is ", np.mean(correlations))



