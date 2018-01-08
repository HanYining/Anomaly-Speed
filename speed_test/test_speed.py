import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import unittest
from speed_cal import *

sample_PICs = read_pics("speed_test/samplePIC/*.jpg")

class Test_iTree(unittest.TestCase):

    def test_read_samplePIC(self):

        # test two sample pictures.
        self.assertTrue(len(sample_PICs) == 2)
        # RGB three colors here
        self.assertTrue(sample_PICs[0].shape[2]== 3)

    def test_split_tile(self):
        tile_u, tile_l = split_tile([[0,10],[10,20]], sample_PICs)
        self.assertEqual(len(tile_u), len(tile_u), 2)
        self.assertEqual(tile_u[0].shape[0], tile_l[0].shape[0], 10)

    def test_get_anomaly_ID(self):
        sample_scores = [random.uniform(0,1) for _ in range(1000)]
        anomaly_id = get_anomaly_frameId(sample_scores, 0.7)
        self.assertTrue(len(anomaly_id) == sum([1 for val in sample_scores if val > 0.7]))

    def test_appear_upper(self):
        # giving a sample trajectory of a moving car,
        # see if the resulting time-location dictionary is accurate.
        anomaly_frame = [[i,i+1,i+2] for i in range(16)]
        locations = frame_tiles(anomaly_frame)
        self.assertTrue(locations[0] == [0])
        self.assertTrue(locations[1] == [0,1])
        for i in range(2,16):
            self.assertTrue(locations[i] == [i-2,i-1,i])
        self.assertTrue(locations[16],[14,15])
        self.assertTrue(locations[17], [15])

    def test_consecutive_seq(self):
        # test for consecutive sequence, with tolerance=1,
        # the function should return all consecutive sequence that is longer
        # than 3, with a pairwise tolerance = 1
        frames = {
            0:[1,2,3,4],
            1:[2,3,5,6],
            2:[2,3],
            3:[2,3,6,7]
        }
        consec_frames = get_consecutive_sequence(frames)
        self.assertTrue(consec_frames=={0:[1,2,3,4],
                                        1:[2,3,4,5,6]})

    def test_mean_loc(self):
        # test the function that returns the mean location at that frame
        frames = {0:[0,1,2,3,4],
                  1:[6,7,8,9,10]}
        self.assertTrue(get_mean_loc(frames) == {0:2,
                                                 1:8})

    def test_instant_speed(self):
        # test a sample frame:location dictionary,
        # see if the instant speed function output the reasonale
        # smoothing speed estimation.
        location = {}
        for i in range(10):
            location[i] = i+1
        self.assertEqual(instant_speed(location), {4:(8-3)/(5)/(300/5600),
                                                   5:(8-3)/(5)/(300/5600)})

    def test_robust_speed(self):
        speed = {1:3,
                 2:4,
                 3:5,
                 4:6}
        multi = 3.6/1.609
        # test multiple smoothing tuning parameter.
        self.assertEqual(robust_speed(speed,5), {1:round(18/4*multi, 2),
                                                 2:round(18/4*multi, 2),
                                                 3:round(18/4*multi, 2),
                                                 4:round(18/4*multi, 2)})
        self.assertEqual(robust_speed(speed,2), {1:round(7/2*multi, 2),
                                                 2:round(7/2*multi, 2),
                                                 3:round(11/2*multi, 2),
                                                 4:round(11/2*multi, 2)})

if __name__ == '__main__':

    unittest.main()
