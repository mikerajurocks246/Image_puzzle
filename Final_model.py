import pandas as pd
import numpy as np
import cv2
import csv
from scipy.stats import pearsonr
from random import randint
from copy import deepcopy

def get_corners(img):
    keys = []
    values = []
    for i in range(6):
        for j in range(6):
            keys.append(tuple([i,j]))
            x1 = (50*i)
            y1 = (50*j)
            x2 = (50*(i+1))
            y2 = (50*(j+1))
            if i == 0:
                x1 = 0
            if j == 0:
                y1 = 0
            values.append([[x1,y1],[x2,y2]])
    return dict(zip(keys, values))

def img_transpose(img):
    arr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            arr[i][j] = img[j][i]
    return arr.astype(int)

def get_edge_vector(img, dictionary):
    keys = []
    values = []

    for i in range(6):
        for j in range(6):
            keys.append(tuple([i, j]))
            lis = dictionary[tuple([i, j])]
            crop = img[lis[0][0]:lis[1][0], lis[0][1]: lis[1][1]]
            #print(crop.shape)
            crop_t = img_transpose(crop)
            top = crop[0]
            bottom = crop[-1]
            left = crop_t[0]
            right = crop_t[-1]
            values.append([top, bottom, left, right])
    return(dict(zip(keys, values)))

def pearson_line(e1, e2):
    ch_r1 = []
    ch_g1 = []
    ch_b1 = []
    ch_r2 = []
    ch_b2 = []
    ch_g2 = []
    for i in range(50):
        ch_r1.append(e1[i][0])
        ch_g1.append(e1[i][1])
        ch_b1.append(e1[i][2])
        ch_r2.append(e2[i][0])
        ch_g2.append(e2[i][1])
        ch_b2.append(e2[i][2])
    corr_r, _ = pearsonr(ch_r1, ch_r2)
    corr_b, _ = pearsonr(ch_b1, ch_b2)
    corr_g, _ = pearsonr(ch_g1, ch_g2)
    sum = corr_r+corr_g+corr_b
    return sum

def get_positions1(edge_dict, comb):
    keys = []
    values = []
    for item in comb:
        keys.append(item)
        closest_ij = []
        min_top = -4
        min_bottom = -4
        min_left = -4
        min_right = -4
        top_ij = ()
        bottom_ij = ()
        left_ij = ()
        right_ij = ()
        for i in range(6):
            for j in range(6):
            
                top1 = edge_dict[item][0]
                bottom1 = edge_dict[item][1]
                left1 = edge_dict[item][2]
                right1 = edge_dict[item][3]
                if item !=(i, j):
                    top2 = edge_dict[(i, j)][0]
                    bottom2 = edge_dict[(i, j)][1]
                    left2 = edge_dict[(i, j)][2]
                    right2 = edge_dict[(i, j)][3]
                    if pearson_line(top1, bottom2)>min_top:
                        top_ij = (i, j)
                        min_top = pearson_line(top1, bottom2)
                    if pearson_line(bottom1, top2)>min_bottom:
                        bottom_ij = (i, j)
                        min_bottom = pearson_line(bottom1, top2)
                    if pearson_line(left1, right2)>min_left:
                        left_ij = (i, j)
                        min_left = pearson_line(left1, right2)
                    if pearson_line(right1, left2)>min_right:
                        right_ij = (i, j)
                        min_right = pearson_line(right1, left2)
        #closest_ij.append()
        lis = [top_ij, bottom_ij, left_ij, right_ij]
        for i in range(len(lis)):
            if lis[i] == ():
                lis[i] = (-1, -1)
        values.append(lis)
    return dict(zip(keys, values))

def check_dict(closest_dict, comb):
    d = closest_dict
    keys = comb
    values = []
    for item in keys:
        final_neighbour = []
        neighbours = closest_dict[item]
        neighbour_top = neighbours[0]
        neighbour_bottom = neighbours[1]
        neighbour_left = neighbours[2]
        neighbour_right = neighbours[3]

        if neighbour_top != (-1,-1):
            if closest_dict[neighbour_top][1] == item:
                final_neighbour.append(neighbour_top)
            else:
                final_neighbour.append((-1,-1))
        else:
            final_neighbour.append((-1,-1))

        if neighbour_bottom != (-1,-1):
            if closest_dict[neighbour_bottom][0] == item:
                final_neighbour.append(neighbour_bottom)
            else:
                final_neighbour.append((-1,-1))
        else:
            final_neighbour.append((-1,-1))
        
        if neighbour_left != (-1,-1):
            if closest_dict[neighbour_left][3] == item:
                final_neighbour.append(neighbour_left)
            else:
                final_neighbour.append((-1,-1))
        else:
            final_neighbour.append((-1,-1))
        
        if neighbour_right != (-1,-1):
            if closest_dict[neighbour_right][2] == item:
                final_neighbour.append(neighbour_right)
            else:
                final_neighbour.append((-1,-1))
        else:
            final_neighbour.append((-1,-1))
        
        values.append(final_neighbour)
    return(dict(zip(keys, values)))

def get_ij(node, matrix):
    #id = [-5,-5]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if node[0]==matrix[i][j][0] and node[1]==matrix[i][j][1]:
                id = [i, j]
                return id
                break
    #return id

def expand_node(node, dictionary ,matrix, in_matrix, not_exp):
    values = dictionary[node]
    id = get_ij(node, matrix)
    #print(node)
    #print(id)
    #print(values)
    if id!=None:
        x = id[0]
        y = id[1]
        if x!=-5 and y!=-5:
            if values[0]!=(-1,-1):
                    matrix[x-1][y] = values[0]
                    in_matrix.append(values[0])
            if values[1]!=(-1,-1):
                    matrix[x+1][y] = values[1]
                    in_matrix.append(values[1])
            if values[2]!=(-1,-1):
                    matrix[x][y-1] = values[2]
                    in_matrix.append(values[2])
            if values[3]!=(-1,-1):
                    matrix[x][y+1] = values[3]
                    in_matrix.append(values[3])
            not_exp.remove(node)
            print("expanded: ", node)
        else:
            not_exp.remove(node)
    else:pass

def set_priority(in_matrix, not_exp):
    x = ()
    for node in not_exp:
        if node in in_matrix:
            x = node
            break
        
    return x     

def find_end(matrix):
    x = 0
    y = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j][0]!=-2 and matrix[i][j][1]!=-2:
                x = i
                y = j
    return [x,y]

def find_start(matrix):
    x = 0
    y = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j][0]!=-2 and matrix[i][j][1]!=-2:
                x = i
                y = j
                return [x,y]
                break

def fetch_positions(final_matrix, comb):
    final_pos = []
    for item in comb:
        final_pos.append(get_ij(item, final_matrix))
    return final_pos


train_faces_csv = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\train\train_faces.csv"
train_landmarks_csv = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\train\train_landmarks.csv"
train_faces = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\train\faces"
train_landamrks = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\train\landmarks"
sample_pred = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\sample_prediction.csv"
test_img = r"C:\Users\ABHINAV\Downloads\mlware23\dataset\test\test_img"
a = [] # keys
b = [] # values
df = pd.read_csv(sample_pred)
# df.shape =  1996 * 37
for count in range(df.shape[0]):
    a.append(df.shape[0])
    img_path = test_img+ chr(92) + df.iloc[count][0]
    img = cv2.imread(img_path)
    dictionary = get_corners(img) # dictionary->corners of each tile
    edge_dict = get_edge_vector(img, dictionary) # edge_dict->stores edges of each tile

    comb = []
    for i in range(6):
        for j in range(6):
            comb.append((i, j))
    closest_dict1 = get_positions1(edge_dict, comb) # closest_dict1->stores the closest pearson pairs
    final_dict = check_dict(closest_dict1, comb) # final_dict->stores the final closest pairs

    matrix = np.empty(shape=(100,100,2))
    matrix.fill(-2)
    not_exp = deepcopy(comb)
    start = not_exp[0]
    matrix[50][50] = start
    in_matrix = [start]
    expand_node(start, final_dict, matrix, in_matrix, not_exp)
    #print(len(not_exp))
    #i = 1
    while len(not_exp)!=0:
        node = set_priority(in_matrix, not_exp)
        #print(node, get_ij(node ,matrix))
        if node == ():
            break
        #print(in_matrix)
        #print(not_exp)
        expand_node(node, final_dict, matrix, in_matrix, not_exp)
        #print(i)
        #i = i+1

    x = find_end(matrix)[0]
    y = find_end(matrix)[1]
    final_matrix = matrix[x-5:x+1,y-5:y+1]

    copy = deepcopy(comb)

    for i in range(final_matrix.shape[0]):
        for j in range(final_matrix.shape[1]):
            if (final_matrix[i][j][0], final_matrix[i][j][1]) in copy:
                copy.remove((final_matrix[i][j][0], final_matrix[i][j][1]))
    x = 0
    for i in range(final_matrix.shape[0]):
        for j in range(final_matrix.shape[1]):
            if final_matrix[i][j][0] < 0 and final_matrix[i][j][1] < 0:
                final_matrix[i][j] = copy[x]
                x = x+1
    #print(fetch_positions(final_matrix, comb))
    b.append(fetch_positions(final_matrix, comb))

opt_dict = dict(zip(a, b))
print(opt_dict)