import glob
import cv2
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import re
import numpy as np
from iforest.iforest import *

def read_pics(files):
    """
    :param files: files representing the picture frames extracted from the cideo
    :return: returns a list of 3-D np.array representing the video
             with the order of np.array as the order for the frames' occurrence in video.
    """
    filenames = glob.glob(files)
    filenames.sort(key = lambda x: int(re.findall(r"[0-9]+", x)[0]))
    frames = []
    for name in filenames:
        frames.append(cv2.imread(name))
    return frames

def split_tile(ranges, frames):
    """
    :param ranges: a list of list with each element represents the upper and lower range of each 
                   lane.
    :param lst_pic: the list of pictures.
    :return: a list of slices of lanes represent the sampling area of our detection.
    """
    return [[ lst[range_i[0]:range_i[1],:,:] for lst in frames] for range_i in ranges]


def feature_extraction(tile, window_len):
    """
    :param tile: tile is one of the lanes partitioned from the pictures,
                 giving a window_lens, the function helps you partition the lane into a set of 
                 datasets representing the bounding box at that lane.
    :param window_len: representing the width of the bounding box used to detect anomaly
                       larger window_len might lead to more robust speed estimation 
                       but can also result in a slower speed update.
    :return: a list of features for each frame is returned. the default would be middle80% range and mean
             for the RGB color.
    """
    windows = [[val, val + window_len] for val in np.arange(0, 640, window_len)]
    data_tile = []
    for window in windows:
        tile_part = [ mat[:,window[0]:window[1],:] for mat in tile]
        data_tile.append(tile_part)
    for i, data in enumerate(data_tile):
        data_tile[i] = np.asarray([np.hstack([np.percentile(frame, 85, axis=(0,1)) - \
                                                 np.percentile(frame, 15, axis=(0,1)),
                                                 np.mean(frame, axis=(0,1)),
                                                 np.median(frame, axis=(0, 1))]) for frame in data])
    return data_tile

def get_scores(tiles,sub_sample_size = 256, forest_size = 100, maximum_depth = 8):
    """
    :param tiles: a list of datasets, which are partitioned by a set of bounding box on that lane.
                  with a number of features for each frame in each bounding box.
    :param sub_sample_size: the random sampling data size used when constructing an isolation tree.
    :param forest_size: number of isolation tree constructed in an isolation forest
    :param maximum_depth: the maximum depth in the construction period of a single isolation tree.
    :return: 
    """
    scores = []
    for tile in tiles:
        scores.append(isolation_forest(tile, sub_sample_size, forest_size,
                                       maximum_depth).evaluate_forest(tile,255))
    return scores

def get_anomaly_frameId(scores, threshold):
    """
    :param scores: a list of list representing the anomaly scores in each frame in each bounding box
    :param threshold: a threshold for an observation to be categorized as anomaly.
    :return: a list of all the bounding box and the frame_id for anomalies in each bounding box
    """
    return [scores.index(val) for val in [val2 for val2 in scores if val2>threshold]]

def tag_anomaly(anomaly_frame, anomaly_range, frames):
    """
    :param anomaly_frame: a list of bounding boxes, with the anomaly observations' frame_id in each bounding box. 
    :param anomaly_range: 
    :param lst_pic: 
    :return: 
    """
    for i in range(0,16):
        anomaly = anomaly_frame[i]
        range_frame = anomaly_range[i]
        for j in anomaly:
            for k in [0,1,2]:
                for m in range_frame[0:2]:
                    frames[j][m, range_frame[2]:range_frame[3], k] = mark_color[k]
                for n in range_frame[2:4]:
                    frames[j][range_frame[0]:range_frame[1], n, k] = mark_color[k]
    return frames

def frame_tiles(anomaly_frame):
    """
    :param anomaly_frame: change the structure from lists of anomalies in tiles
                          to list of anomalies in frames
    :return: a dictionary, with frames which have anomaly tiles as the key and 
             the corresponding id of anomaly tiles in the lists.
    """
    frame_tile = {}
    for i, tile in enumerate(anomaly_frame):
        for frame in tile:
            if frame in frame_tile.keys() and i not in frame_tile[frame]:
                frame_tile[frame].append(i)
            else:
                frame_tile[frame] = [i]
    return frame_tile

def get_consecutive_sequence(frames, tolerance=1):
    # this function should be easy if we know our anomaly detection
    # have 100% accuracy, but it turns out not the case.
    # the car can be partitioned by certain mistakes,
    # which resulted that at certain frame, the tiles represent the car
    # might not be continuous, etc [1,2,4,5], here 3 is a false negative.
    # here in order to give a robust speed estimate
    # we need to track the sequence of anomaly tiles over each frame.
    # to increase the robustness of the longest subsequence in this case
    # I defined a tolerance with default=1 that represent the amount of gap lengths between anomaly tiles
    # and restore the separate tiles like [1,2,4,5] back into [1,2,3,4,5]
    """
    :param frames: a dictionary with frame id as keys and anomaly tiles as values.
    :return: a new dictionary with frame id as keys and corresponding consecutive anomaly tiles as values 
    """
    frames2 = {}
    for frame in frames.keys():
        sequence = frames[frame]
        if len(sequence)<3:
            continue
        left = sequence[0]
        right = sequence[0]
        for i in range(len(sequence)-1):
            if sequence[i]+1+tolerance >= sequence[i+1]:
                right = sequence[i+1]
            elif right-left<2:
                right = sequence[i+1]
                left = sequence[i+1]
            else:
                break
        if right-left>=2:
            frames2[frame] = [i for i in range(left,right+1)]
    return frames2

def get_mean_loc(frames):
    """
    :param frames: a dictionary with the key as frames and the value as anomaly tiles 
    :return: a dictionary with the key as frames and the value as the mean location
             of the anomaly tiles 
    """
    # now we get a hash table contain the anomaly tiles in each frame
    # ideally the mean location of the car at that time could be
    # represented by the mean of that sequence.
    for frame in frames.keys():
        frames[frame] = np.mean(frames[frame])
    return frames

def instant_speed(anomaly_frames):
    """
    :param anomaly_frames: a dictionary with frames as keys and the anomaly tiles ID as values
    :return: a dictionary with frames as keys and a smoothing speed estimation as values
    """
    # here I estimate the speed at a frame by look forward 20 frames take the
    # average loc of that time period
    # and look back ten frames take the average loc of that time period
    speed = {}
    frames = sorted(anomaly_frames.keys())
    for frame in frames:
        upper, lower = [], []
        upper_loc, lower_loc = [], []
        for i in range(frame, frame+20):
            if i in frames:
                upper.append(anomaly_frames[i])
                upper_loc.append(i)
        for j in range(frame-20, frame+1):
            if j in frames:
                lower.append(anomaly_frames[j])
                lower_loc.append(j)
        if len(upper_loc) >= 5 and len(lower_loc)>=5:
            # here 300 means 300 second, totally 5 mins video, 5600 means total separated frames.
            speed[frame] = abs((np.mean(upper) - np.mean(lower)))/(np.mean(upper_loc)-np.mean(lower_loc))/(300/5600)
    return speed

def robust_speed(speed_est, average_len):
    """
    :param speed_est: a dictionary with frame-id as the keys and speed as the value 
    :return: a dictionary with frame-id as the keys and robust speed estimator as the values 
    """
    # here since estimating a single speed at each frame can be highly unreliable
    # also the fast change of the speed estimation make it hard to visualize
    # here I take an average speed in a 10 frame period.
    frames = list(speed_est.keys())
    i = 0
    robust = {}
    while i < len(frames):
        sum_speed = 0
        cnt = 0
        inrange = []
        for j in range(0,average_len):
            if i+j<len(frames) and frames[i+j] <= frames[i]+average_len:
                sum_speed += speed_est[frames[i+j]]
                cnt += 1
                inrange.append(i+j)
        for val in inrange:
            # the distance of the road in the video is 17 meters
            # change the speed from m/s to mile/hour
            robust[frames[val]] = round(sum_speed/cnt * 3.6/1.609,2)
        i = max(inrange)+1
    return robust

# use the robust_speed estimation to tag the video
def add_tag_to_pic(pictures, speed_upper, speed_lower):
    # tag the corresponding pictures.
    pics = []
    for i,pic in enumerate(pictures):
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(pic, "Speed", (0, 180), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if i in speed_upper.keys():
            cv2.putText(pic, str(round(speed_upper[i],1)) + "mph", (80, 180), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(pic, "Speed", (0, 280), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if i in speed_lower.keys():
            cv2.putText(pic, str(round(speed_lower[i],1)) + "mph", (80, 280), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        pics.append(pic)
    return pics

if __name__ == '__main__':

    frames = read_pics("image/*.jpg")

    tile1,tile2 = split_tile([[135,215],[245,310]], frames)

    tiles_upper = feature_extraction(tile1, 40)
    tiles_lower = feature_extraction(tile2, 40)

    upper_scores = get_scores(tiles_upper)
    lower_scores = get_scores(tiles_lower)

    # choose anomaly score to be 0.6
    anomaly_frame_upper = [get_anomaly_frameId(scores, 0.6) for scores in upper_scores]
    anomaly_range_upper = [[135,215,i,i+40] for i in range(0,640,40)]
    anomaly_range_upper[-1][3] = 639

    anomaly_frame_lower = [get_anomaly_frameId(scores, 0.6) for scores in lower_scores]
    anomaly_range_lower = [[245,310,i,i+40] for i in range(0,640,40)]
    anomaly_range_lower[-1][3] = 639

    # add green bounding box to moving cars
    mark_color = {0:0,1:252,2:0}
    frames = tag_anomaly(anomaly_frame_upper, anomaly_range_upper, frames)
    frames = tag_anomaly(anomaly_frame_lower, anomaly_range_lower, frames)

    appear_upper = frame_tiles(anomaly_frame_upper)
    appear_lower = frame_tiles(anomaly_frame_lower)

    appear_upper = get_consecutive_sequence(appear_upper)
    appear_lower = get_consecutive_sequence(appear_lower)

    location_upper = get_mean_loc(appear_upper)
    location_lower = get_mean_loc(appear_lower)

    instant_speed_upper = instant_speed(location_upper)
    instant_speed_lower = instant_speed(location_lower)

    robust_upper = robust_speed(instant_speed_upper, average_len=20)
    robust_lower = robust_speed(instant_speed_lower, average_len=20)

    frames = add_tag_to_pic(frames, robust_upper,robust_lower)

    videoWriter = cv2.VideoWriter("speed_estimation.avi", VideoWriter_fourcc('X','V','I','D'), 20, (640,368))
    for image_modified in frames:
        videoWriter.write(image_modified)

    videoWriter.release()
