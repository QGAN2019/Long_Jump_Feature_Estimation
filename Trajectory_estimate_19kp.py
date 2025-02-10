# %%

import numpy as np
import pandas as pd
import re
import os
from matplotlib import pyplot as plt
import csv
from utils import *
import argparse


def estimate(athleteNo, exp, args, printing=False, displaying=False):
  #athleteNo = "04"
  #exp = "results351C"

  gt = pd.read_csv(args.feature_gt, index_col=0)
  #print(gt.keys())

  # Defining constants and parameters
  PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], 
          [0, 5], [5, 6], [6, 7], [7, 8], 
          [0, 9], [9, 10],[10, 11], [11, 12], 
          [10, 13], [13, 14], [14, 15],
          [10, 16], [16, 17], [17, 18], [10, 12]]
  LINKS = ['RPelvis','RThigh','RCalf','RFoot',
          'LPelvis','LThigh','LCalf','LFoot',
          'Spine2', 'Spine1','Neck', 'Head',
          'LClavicle','LUpperArm','LForearm',
          'RClavicle','RUpperArm','RForearm', 'NeckHead']

  Nkp = 19

  Start_Frame = 0 #5
  End_Frame = -1 #89

  # data_root = "/home/qgan/projects/[Paper]FeatureEstimate_HSI24/Datasets/Long_jump/data/experiments"  # './yolov6_vitpose_MHFormer351'
  #data_root = './yolov6_vitpose_MHFormer351'

  input_path =  os.path.join(args.data_root, athleteNo, exp)  # './yolov3_hrnet_MHFormer351/01'
  # output_path = os.path.join(input_path, 'results')
  output_path = args.output_path # "/home/qgan/projects/[Paper]FeatureEstimate_HSI24/Datasets/Long_jump/data/"
  os.makedirs(output_path, exist_ok=True)

  f = open(os.path.join(input_path,"out_pose3D.txt"), "r")
  data = f.readlines()
  poses = []
  for i,d in enumerate(data):
    if i%Nkp==0:
      keypoints = []
    kp =  re.sub("\s+", " ",d.strip("[ \n]")).split()
    kp = [float(i) for i in kp]
    keypoints.append(kp)
    if i%Nkp==(Nkp-1):
      poses.append(keypoints)
  poses_np = np.array(poses[Start_Frame:End_Frame])
  #print(len(poses), poses_np.shape)

  # Normalization
  Height = float(gt[str(int(athleteNo))+" "]["Height"])

  link_lengths = dict()
  for idx,link in enumerate(LINKS):
    link_lengths[link] = np.linalg.norm(poses_np[:,PAIRS[idx][0],:] - poses_np[:,PAIRS[idx][1],:],axis=1)

  bone_lengths = dict()
  for link in link_lengths.keys():
    m = np.mean(link_lengths[link])
    bone_lengths[link] = m
    s = np.std(link_lengths[link])
    #print(link, ': ', '\t', m, '\t', s, '\t', s/m*100, '%')
  #print(bone_lengths)

  H =(bone_lengths['RFoot']+bone_lengths['LFoot'])/4+(bone_lengths['RThigh']+bone_lengths['RCalf']+bone_lengths['LThigh']+bone_lengths['LCalf'])/2+bone_lengths['Spine1']+bone_lengths['Spine2']+bone_lengths['NeckHead'] # bone_lengths['Neck']+bone_lengths['Head']
  ratio = Height/H

  Total_lengths = []
  for i in range(poses_np.shape[0]):
    total_length = 0
    for link in LINKS:
      total_length += link_lengths[link][i]
    #print(i,':',total_length)
    Total_lengths.append(total_length)
  #print(np.mean(Total_lengths),np.std(Total_lengths))

  poses_norm = np.zeros_like(poses_np)
  for i in range(poses_np.shape[0]):
    poses_norm[i,...] = poses_np[i,...]/Total_lengths[i]*np.mean(Total_lengths)*ratio
    #poses_norm[i,:,2] = poses_norm[i,:,2] - np.min(poses_norm[i,:,2])

  """for i in range(poses_norm.shape[0]):
    print(poses_norm[i,:,2])"""

  try:
    f = open(os.path.join(args.data_root, athleteNo, "contact.txt"), "r")
    data = f.readlines()
    contacts = []
    for d in data:
      ct =  int(d.strip(" \n"))
      contacts.append(ct)
    #print(len(contacts))
    contacts_np = np.array(contacts[Start_Frame:End_Frame])
  except:
    contacts_np = np.load(os.path.join(args.data_root, athleteNo,"contacts.npy"), allow_pickle=True)
    contacts_np = contacts_np[Start_Frame:End_Frame]

  # Drop beginning sequences if they are off-ground or only 1 kp on-ground
  # Drop ending sequences if they are on-ground
  idx_start = 0
  while contacts_np[idx_start] * contacts_np[idx_start+1] == 0:
    idx_start += 1

  idx_end = 0
  while contacts_np[idx_end-2] == 2:
    idx_end -= 1

  if idx_end == 0:
    poses_cut = poses_norm[idx_start:,...]
    contacts_cut = contacts_np[idx_start:]
  else:
    poses_cut = poses_norm[idx_start:idx_end,...]
    contacts_cut = contacts_np[idx_start:idx_end]

  #print(contacts_np.shape, poses_norm.shape)
  #print(poses_cut.shape, contacts_cut.shape, idx_start, idx_end)
  """for i in range(poses_cut.shape[0]):
    print(poses_cut[i,:,2])"""

  phases = contact2phase(contacts_cut)
  if printing: 
    print(phases)

  CoMs = calcCoM(poses_cut, gender=1)
  #print(CoMs.shape)
  CoMs_np = np.array(CoMs)
  ###plt.plot(CoMs_np[:,2])

  pose_global, CoM_global = calcTraj(poses_cut,phases,CoMs)
  np.save(os.path.join(output_path, 'global_3d', str(athleteNo)+'.npy'), pose_global)

  """for i in range(pose_global.shape[0]):
    print(pose_global[i,:,2])"""

  fig1 = plt.figure()
  plt.scatter(CoM_global[:,0], CoM_global[:,1])
  plt.savefig(os.path.join(output_path,"CenterOfMass_birdview.png"))

  fig2 = plt.figure()
  plt.plot(CoM_global[:,2],'o-')
  plt.savefig(os.path.join(output_path,"CenterOfMass_sideview.png"))

  fig3 = plt.figure(figsize=(20,2.5))
  poses = pose_global
  step = 2
  for i in range(0,poses.shape[0],step):
    p3d = poses[i]
    plt.scatter(p3d[:, 0], p3d[:, 2])
    for pair in PAIRS[:18]:
      plt.plot(p3d[pair, 0], p3d[pair, 2])

  plt.savefig(os.path.join(output_path,"Trajectory_sideview.png"))


  fig4 = plt.figure(figsize=(20,5))
  poses = pose_global
  step = 3
  for i in range(0,poses.shape[0],step):
    p3d = poses[i]
    plt.scatter(p3d[:, 0], p3d[:, 1])
    for pair in PAIRS[:18]:
      plt.plot(p3d[pair, 0], p3d[pair, 1])

  plt.savefig(os.path.join(output_path,"Trajectory_birdview.png"))

  features = Features(CoM_global, pose_global, phases, dt)
  features.calc_features()

  # Save results
  results_path = os.path.join(output_path,"results.csv")
  #save_dict(features.features, results_path)
  w = csv.writer(open(results_path, "w"))
  for key in Feature_List:
      w.writerow([key, features.features[key]])

  if printing:
    for ft in features.features:
      print(ft,":", features.features[ft])

    print(eval_feature_estimation(features.features, list(gt[athleteNo][4:])))

  # gt = pd.read_csv('Ground_truth_features_2018.csv')

  if displaying:
    plt.show()

################################################################################
athleteNo_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
                  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                  "21", "22", "23", "24", "25", "26"]

"""athleteNo_list = ["13", "14", "21"]"""

exp_list = ["results351C"] # ["results351C", "results351N", "results81C"]

for ath in athleteNo_list:
  for exp in exp_list:
    estimate(ath, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='input video')
    parser.add_argument('--output_path', type=str, default='./output', help='output 2D keypoints folder')
    parser.add_argument('--feature_gt', type=str, default='./ground_truth_features_17_18.csv', help='the dataset used (h36m or 3dHP)')
    
    args = parser.parse_args()

    athleteNo_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
                      "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                      "21", "22", "23", "24", "25", "26"]

    """athleteNo_list = ["13", "14", "21"]"""

    exp_list = ["results351C"] # ["results351C", "results351N", "results81C"]

    for ath in athleteNo_list:
      for exp in exp_list:
        estimate(ath, exp, args)