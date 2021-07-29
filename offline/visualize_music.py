"""Visualization tools."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shutil
import os
import librosa
from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import collections
import pandas as pd

join=os.path.join

def save_video(songname, songlen, total_frames, dance_images_dir, output):
    """Make video from given frames. Add audio appropriately."""
    num_steps_by_len = 30
    print("the fps is ", num_steps_by_len)
    print("the song len is ", songlen)
    dance_images = join(dance_images_dir, 'image%06d.jpg') 

    p = subprocess.Popen(['ffmpeg', '-r', str(num_steps_by_len), '-i', dance_images,
                        '-i', songname, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-y', '-strict', '-2', output])

    p.wait()

def generate_mp3(fname):
    inp_fname = fname
    out_fname = fname.replace(".mid", ".mp3")
    p = subprocess.Popen(['timidity', inp_fname, '-Ow', '-o', out_fname])
    p.wait()
    return out_fname

def generate_wav(fname):
    inp_fname = fname
    out_fname = fname.replace(".mid", ".wav")
    p = subprocess.Popen(['timidity', inp_fname, '-Ow', '-o', out_fname])
    p.wait()
    return out_fname

def save_matrices(dance_matrix, music_matrix, duration, fname):
    """Save music and dance matrices."""
    d = int(duration)
    print(dance_matrix.shape[0], duration, d)

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(dance_matrix, cmap='gray')
    plt.savefig(fname + 'dance.png', bbox_inches='tight')
    plt.close()

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(music_matrix, cmap='gray')
    plt.savefig(fname + 'music.png', bbox_inches='tight')
    plt.close()


def save_dance_matrix(dance_matrix, duration, fname):

    d = int(duration) + 1

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, dance_matrix.shape[0], dance_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(dance_matrix, cmap='gray')
    plt.savefig(fname + 'dance.png', bbox_inches='tight')
    plt.close()
    
def save_dance(states, visfolder, songname, duration, num_steps):
    """Save dance."""
    # Make folder if not already exists
    if not os.path.exists('./plots/'):
        os.makedirs('plots/')

    # Delete old items
    print("Starting file deletions")
    for item in os.listdir('./plots/'):
        delfile = os.path.join('./plots/', item)
        os.remove(delfile)
    print("File deletions complete")

    # Create dance video
    print("Creating dance video frames")
    c = 0
    for i, state in enumerate(states):
        # ****************** stick figure agent ******************
        shutil.copy(visfolder + "/" + str(state+1) + '.png', 'plots/' + str(i+1) + '.png')
        c += 1

    # Save video
    save_video('./plots', songname, duration, num_steps, songname + '.mp4')

def save_music(fname, fps, states, baseline_indices):

    melody_list = [70 if val == 100 else val for val in states]

    # fps = 30
    frames_per_note = 6
    num_notes = len(melody_list)
    duration_of_one_note = frames_per_note / fps
    melody_list_modified = []
    curr_note = melody_list[0]
    note_duration = []
    count = 0
    for idx,note in enumerate(melody_list):
        if idx == 0:
            curr_note = note
            count = 1
            continue
        else:
            if note == curr_note:
                count+=1
            else:
                melody_list_modified.append(curr_note)
                note_duration.append(count)
                curr_note = note
                count = 1
    melody_list_modified.append(curr_note)
    note_duration.append(count)
    print(melody_list)
    print(melody_list_modified)
    print(note_duration)
    pd_dict = collections.defaultdict(list)
    pd_dict_baseline = collections.defaultdict(list)
    start_time = 0
    for idx,note in enumerate(melody_list_modified):
        pd_dict['start_time'].append(start_time)
        end_time = start_time + note_duration[idx]*duration_of_one_note
        start_time = end_time
        pd_dict['end_time'].append(end_time)
        pd_dict['bottom'].append(note - 0.4)
        pd_dict['top'].append(note + 0.4)

    for idx,note in enumerate(melody_list):
        if idx in baseline_indices:
            pd_dict_baseline['start_time'].append(idx*duration_of_one_note)
            pd_dict_baseline['end_time'].append((idx+1)*duration_of_one_note)
            pd_dict_baseline['bottom'].append(note - 0.4)
            pd_dict_baseline['top'].append(note + 0.4)

    source = ColumnDataSource(pd.DataFrame(pd_dict))
    source_baseline = ColumnDataSource(pd.DataFrame(pd_dict_baseline))

    plot = figure(plot_width=400, plot_height=400, title= str(fname.split("/")[-1].split(".")[0]))
    plot.xaxis.axis_label = 'time (sec)'
    plot.yaxis.axis_label = 'note'
    plot.quad(top='top', bottom='bottom', left='start_time', right='end_time',
              line_color='black', fill_color="#B3DE69",source=source)
    plot.quad(top='top', bottom='bottom', left='start_time', right='end_time',
              line_color='black', fill_color="#F1084B",source=source_baseline)
    # show(plot)
    # export_png(plot)
    output_file(fname)
    save(plot)


if __name__ == '__main__':
    songpath = "vis_num_steps_20/dancing_person_20/output_midi_9_baseline_steps_50_fps_30.mp3"
    # songpath = "vis_num_steps_20/dancing_person_20/output_midi_31_baseline_steps_25.mp3"
    y, sr = librosa.load(songpath)    # default sampling rate 22050
    duration = librosa.get_duration(y=y, sr=sr)
    foldername = "vis_num_steps_20/dancing_person_20/"
    # output = foldername + "movie_31_ftm_state_steps_50_correct.mp4"
    output = foldername + "movie_9_ftm_state_steps_50_fps_30.mp4"
    # output = foldername + "movie_31_ftm_state_steps_25.mp4"
    save_video(foldername, songpath, duration, 100, output)
