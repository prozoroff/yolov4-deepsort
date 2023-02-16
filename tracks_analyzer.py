import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from correct_distortion import correct_image, correct_point

IMAGE_H = 1520
IMAGE_W = 2592
GRID_SIZE = 512
MAX_SPEED = 7

src = np.float32([[0, 225], [IMAGE_W, 225], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
dst = np.float32([[960, 225], [1452, 225], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

M.dump('./outputs/transform.dat')

pluck = lambda dict, *args: (dict[arg] for arg in args)

def trasnformPoint(p):
    px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
    return (int(px), int(py))

def trasnformPointInv(p):
    px = (Minv[0][0]*p[0] + Minv[0][1]*p[1] + Minv[0][2]) / ((Minv[2][0]*p[0] + Minv[2][1]*p[1] + Minv[2][2]))
    py = (Minv[1][0]*p[0] + Minv[1][1]*p[1] + Minv[1][2]) / ((Minv[2][0]*p[0] + Minv[2][1]*p[1] + Minv[2][2]))
    return (int(px), int(py))

def normalize_tracks(tracks):
    tracks_normalized = {}
    for key in tracks.keys():
        tracks_normalized[key] = []
        for (point, frame) in tracks[key]:
            corrected_point = correct_point(point)
            transformed_point = trasnformPointInv(corrected_point)
            tracks_normalized[key].append((transformed_point, frame))
    return tracks_normalized

def get_bounds(self):
    left_top = trasnformPointInv((0,0))
    right_top = trasnformPointInv((IMAGE_W,0))
    left_bottom = trasnformPointInv((0,IMAGE_H))
    right_bottom = trasnformPointInv((IMAGE_W,IMAGE_H))
    return (
        np.min([left_top[0], left_bottom[0]]),
        np.min([left_top[1], right_top[1]]),
        np.max([right_top[0], right_bottom[0]]),
        np.max([left_bottom[1], right_bottom[1]])
    )

def fill_polly_alpha(img, points, color, alpha):
    overlay = img.copy()
    cv2.fillPoly(overlay, np.int32([points]), color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def number_to_color(num, max):
    fromR = 255
    fromG = 0
    fromB = 0

    toR = 0
    toG = 255
    toB = 0

    deltaR = round((toR - fromR) / max)
    deltaG = round((toG - fromG) / max)
    deltaB = round((toB - fromB) / max)

    R = fromR + num * deltaR
    G = fromG + num * deltaG
    B = fromB + num * deltaB

    return (B, G, R)

def get_speed(px_per_frame):
    WIDTH_IN_METERS = 11
    FRAMES_PER_SECOND = 14.4
    return (px_per_frame / IMAGE_W) * WIDTH_IN_METERS * FRAMES_PER_SECOND * 3.6
    
class Grid:
    def __init__(self, size):
        self.mat = [[0 for x in range(size)] for x in range(size)]
        self.max_len = 0

    def set(self, coord, velocity):
        item = self.mat[coord[0]][coord[1]]
        if item == 0:
            item = {'average': 0, 'data': []}
        item['data'].append(velocity)
        item['average'] = np.average(item['data'])
        self.max_len = max(self.max_len, len(item['data']))
        self.mat[coord[0]][coord[1]] = item

    def get(self, coord):
        return self.mat[coord[0]][coord[1]]


img = correct_image(cv2.imread('./data/images/frame.png'))

tracks = {}
with open('./outputs/tracks_ice.pkl', 'rb') as f:
    tracks = pickle.load(f)

tracks_normalized = normalize_tracks(tracks)
min_x, min_y, max_x, max_y = get_bounds(tracks_normalized)

grid = Grid(GRID_SIZE + 1)
step_x = (max_x - min_x) / GRID_SIZE
step_y = (max_y - min_y) / GRID_SIZE

for key in tracks_normalized.keys():
    points = tracks_normalized[key]
    for i in range(len(points)):
        if i > 0:
            cur_point = points[i][0]
            prev_point = points[i-1][0]
            frames_between = points[i][1] - points[i-1][1] 
            velocity = math.sqrt(pow(cur_point[0] - prev_point[0], 2) + pow(cur_point[1] - prev_point[1], 2)) / frames_between
            coord = (int((cur_point[0] - min_x)/step_x), int((cur_point[1] - min_y)/step_y))
            coord = (max(coord[0], 0), max(coord[1], 0))
            coord = (min(coord[0], GRID_SIZE), min(coord[1], GRID_SIZE))
            if velocity > 1:
                grid.set(coord, velocity)

x = min_x
y = min_y

for j in range(GRID_SIZE):
    for i in range(GRID_SIZE):
        item = grid.get((i, j))
        if item != 0:
            t_l = trasnformPoint([min_x + int(i * step_x), min_y + int(j * step_y)])
            t_r = trasnformPoint([min_x + int(i * step_x) + int(step_x), min_y + int(j * step_y)])
            b_l = trasnformPoint([min_x + int(i * step_x), min_y + int(j * step_y) + int(step_y)])
            b_r = trasnformPoint([min_x + int(i * step_x) + int(step_x), min_y + int(j * step_y) + int(step_y)])
            points = np.array([t_l, t_r, b_r, b_l])
            speed = get_speed(item['average'])
            alpha = math.sqrt(len(item['data']) / grid.max_len)
            color = number_to_color(speed, MAX_SPEED)
            fill_polly_alpha(img, np.int32([points]), color, alpha)

while x <= max_x:
    cv2.line(img, trasnformPoint((x, min_y)), trasnformPoint((x, max_y)), (255,255,255), 2)
    x += step_x

while y <= max_y:
    cv2.line(img, trasnformPoint((min_x, y)), trasnformPoint((max_x, y)), (255,255,255), 2)
    y += step_y

plt.imshow(img)
cv2.imwrite('./outputs/images/blank_ice.png', img)
plt.show()


