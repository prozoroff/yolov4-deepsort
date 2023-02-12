import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

IMAGE_H = 1520
IMAGE_W = 2592
GRID_SIZE = 200
MAX_SPEED = 7

src = np.float32([[0, 225], [IMAGE_W, 225], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
dst = np.float32([[860, 225], [1552, 225], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
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
            transformed_point = trasnformPointInv(point)
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

    return (R, G, B)

def get_speed(px_per_frame):
    WIDTH_IN_METERS = 7
    FRAMES_PER_SECOND = 24
    return (px_per_frame / IMAGE_W) * WIDTH_IN_METERS * FRAMES_PER_SECOND * 3.6
    
class Grid:
    def __init__(self, size):
        self.mat = [[0 for x in range(size)] for x in range(size)]

    def set(self, coord, velocity):
        item = self.mat[coord[0]][coord[1]]
        if item == 0:
            item = {'average': 0, 'data': []}
        item['data'].append(velocity)
        item['average'] = np.average(item['data'])
        self.mat[coord[0]][coord[1]] = item

    def get(self, coord):
        return self.mat[coord[0]][coord[1]]


img = cv2.imread('./data/images/frame.png')

tracks = {'person-1': [((152, 761), 3), ((156, 764), 4), ((159, 762), 5), ((165, 762), 6), ((168, 762), 7), ((172, 762), 8), ((176, 765), 9), ((182, 773), 10), ((184, 772), 11), ((188, 771), 12), ((192, 769), 13), ((194, 766), 14), ((197, 766), 15), ((204, 795), 16), ((208, 804), 17), ((211, 807), 18), ((218, 805), 19), ((225, 804), 20), ((229, 807), 21), ((239, 806), 22), ((244, 809), 23), ((250, 814), 24), ((254, 823), 25), ((256, 827), 26), ((259, 837), 27), ((264, 844), 28), ((266, 850), 29), ((268, 854), 30), ((275, 853), 31), ((280, 854), 32), ((286, 857), 33), ((292, 857), 34), ((300, 861), 35), ((305, 865), 36), ((308, 868), 37), ((310, 870), 38), ((314, 879), 39), ((322, 892), 40), ((327, 896), 41), ((330, 906), 42), ((335, 909), 43), ((338, 911), 44), ((344, 913), 45), ((350, 922), 46), ((356, 929), 47), ((359, 932), 48), ((365, 938), 49), ((369, 939), 50), ((379, 955), 51), ((386, 963), 52), ((390, 963), 53), ((393, 974), 54), ((401, 993), 55), ((405, 1002), 56), ((411, 1007), 57), ((418, 1007), 58), ((424, 1008), 59), ((436, 1020), 60), ((449, 1027), 61), ((456, 1029), 62), ((462, 1034), 63), ((467, 1043), 64), ((474, 1058), 65), ((482, 1065), 66), ((488, 1072), 67), ((492, 1073), 68), ((500, 1096), 69), ((509, 1107), 70), ((514, 1108), 71), ((525, 1117), 72), ((542, 1128), 73), ((552, 1134), 74), ((558, 1137), 75), ((564, 1140), 76), ((573, 1148), 77), ((581, 1165), 78), ((591, 1181), 79), ((596, 1180), 80), ((602, 1193), 81), ((611, 1203), 82), ((617, 1208), 83), ((625, 1216), 84), ((657, 1240), 86), ((675, 1246), 87), ((685, 1251), 88), ((696, 1259), 89), ((702, 1263), 90), ((709, 1285), 91), ((719, 1294), 92), ((746, 1358), 94), ((754, 1366), 95), ((767, 1369), 96), ((778, 1381), 97), ((788, 1366), 98), ((811, 1376), 99), ((837, 1387), 100), ((862, 1413), 101), ((874, 1411), 102), ((885, 1421), 103), ((892, 1439), 104), ((906, 1486), 105), ((917, 1505), 106), ((927, 1515), 107), ((948, 1518), 108), ((967, 1520), 109), ((981, 1532), 110), ((1004, 1525), 111), ((1020, 1535), 112), ((1048, 1523), 113), ((1064, 1531), 114), ((2514, 1086), 156), ((2523, 1071), 157), ((2520, 1065), 158), ((2551, 1054), 159)], 'person-2': [((2499, 1075), 47), ((2485, 1077), 48), ((2475, 1078), 49), ((2424, 1064), 51), ((2403, 1061), 52), ((2377, 1058), 53), ((2365, 1054), 54), ((2348, 1048), 55), ((2332, 1045), 56), ((2316, 1043), 57), ((2283, 1045), 58), ((2263, 1043), 59), ((2244, 1041), 60), ((1960, 1032), 75), ((1937, 1022), 76), ((1920, 1017), 77), ((1880, 1015), 78), ((1865, 1017), 79), ((1851, 1015), 80), ((1834, 1010), 81), ((1811, 1008), 82), ((1791, 1007), 83), ((1761, 1011), 84), ((1738, 1012), 85), ((1720, 1016), 86), ((1699, 1016), 87), ((1652, 1003), 89), ((1622, 1006), 90), ((1602, 1000), 91), ((1581, 992), 92), ((1552, 989), 93), ((1529, 986), 94), ((1504, 974), 95), ((1470, 985), 96), ((1456, 992), 97), ((1440, 990), 98), ((1418, 989), 99), ((1348, 983), 102), ((1325, 982), 103), ((1239, 965), 107), ((1213, 962), 108), ((1188, 948), 109), ((1175, 944), 110), ((1157, 947), 111), ((1126, 943), 112), ((1104, 940), 113), ((1084, 936), 114), ((1061, 939), 115), ((1039, 937), 116), ((945, 930), 121), ((924, 928), 122), ((910, 925), 123), ((889, 923), 124), ((736, 904), 132), ((717, 894), 133), ((698, 891), 134), ((681, 891), 135), ((671, 893), 136), ((653, 891), 137), ((613, 879), 139), ((602, 875), 140), ((588, 870), 141), ((572, 870), 142), ((558, 868), 143), ((540, 867), 144), ((517, 853), 145), ((509, 847), 146), ((503, 848), 147), ((487, 844), 148), ((477, 845), 149), ((462, 842), 150), ((440, 835), 151), ((431, 834), 152), ((422, 826), 153), ((412, 823), 154), ((399, 820), 155), ((388, 807), 156), ((375, 802), 157), ((357, 804), 158), ((343, 800), 159), ((328, 804), 161), ((316, 803), 162), ((304, 801), 163), ((288, 795), 166), ((283, 795), 167), ((272, 792), 168), ((254, 782), 169), ((243, 778), 170), ((229, 781), 171), ((221, 777), 172), ((216, 776), 173), ((206, 778), 174), ((194, 770), 175), ((184, 768), 176), ((171, 765), 180), ((163, 752), 181), ((155, 747), 182), ((147, 743), 183), ((135, 750), 184), ((128, 748), 185), ((120, 746), 186), ((78, 726), 194), ((72, 723), 195)], 'person-7': [((2373, 1050), 166), ((2359, 1050), 167), ((2336, 1046), 168), ((2309, 1047), 169), ((2296, 1044), 170), ((2278, 1041), 171), ((2262, 1033), 172), ((2246, 1030), 173), ((2016, 874), 201), ((2009, 868), 202), ((2017, 875), 203), ((2020, 876), 204), ((2016, 872), 205)]}
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
            grid.set(coord, velocity)

x = min_x
y = min_y

while x <= max_x:
    cv2.line(img, trasnformPoint((x, min_y)), trasnformPoint((x, max_y)), (255,255,255), 1)
    x += step_x

while y <= max_y:
    cv2.line(img, trasnformPoint((min_x, y)), trasnformPoint((max_x, y)), (255,255,255), 1)
    y += step_y

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
            alpha = len(item['data']) / 10
            color = number_to_color(speed, MAX_SPEED)
            fill_polly_alpha(img, np.int32([points]), color, alpha)

plt.imshow(img)
plt.show()


