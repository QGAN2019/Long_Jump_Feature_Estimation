import numpy as np
from scipy.signal import savgol_filter,argrelextrema
import math
import csv

"""
'Hip' #0
'RHip' #1
'RKnee' #2
'RFoot' #3
'RToe' #4
'LHip' #5
'LKnee' #6
'LFoot' #7
'LToe' #8
'Spine' #9
'Thorax' #10
'Neck/Nose' #11
'Head' #12
'LShoulder' #13
'LElbow' #14
'LWrist' #15
'RShoulder' #16
'RElbow' #17
'RWrist' #18
"""

Frame_rate = 25
dt = 1 / Frame_rate
G = 9.8

Feature_List = ["Overall - effective distance",
                "Approach - contact time - last",
                "Approach - contact time - 2nd",
                "Approach - contact time - 3rd",
                "Approach - flight time - last",
                "Approach - flight time - 2nd",
                "Approach - flight time - 3rd",
                "Approach - step length - last",
                "Approach - step length - 2nd",
                "Approach - step length - 3rd",
                "Approach - velocity - last",
                "Approach - velocity - 2nd",
                "Approach - velocity - 3rd",
                "TakeOff - horizontal velocity",
                "TakeOff - vertical velocity",
                "TakeOff - horizontal velocity loss",
                "TakeOff - velocity",
                "TakeOff - angle",
                "TakeOff - body inclination at TD",
                "TakeOff - body inclination at TO",
                "TakeOff - trunk inclination at TD",
                "TakeOff - trunk inclination at TO",
                "TakeOff - leading thigh angle at TO",
                "TakeOff - leading thigh angular velocity",
                "TakeOff - knee angle at TD",
                "TakeOff - knee angle (minimum)",
                "TakeOff - knee angle motion range",
                "TakeOff - knee angle angular velocity",
                "TakeOff - CoM lowering",
                "Landing - knee angle",
                "Landing - trunk angle",
                "Landing - landing distance"]


# Mass percentage for men: (Reference: 'Center of Mass of Human’s Body Segments')
# Head(+neck): 6.94, Trunk: 43.46, Upper/Mid/Lower-Trunk: 15.96/16.33/11.17
# UpperArm(single): 2.71, Forearm(single): 1.62, Hand(single): 0.61
# Thigh(single): 14.16, Shank(single): 4.33, Foot(single): 1.37
# Similar for women

def calcCoM(kp,gender=1):
  '''
  kp:17x3
  gender: 0--Female, 1--Male
  CenterOfMass
  Centers: head+neck(0),trunk(1),
       upperarm_l(2),forearm_l(3),hand_l(4),
       upperarm_r(5),forearm_r(6),hand_r(7),
       thigh_l(8),shank_l(9),foot_l(10),
       thigh_r(11),shank_r(12),foot_r(13)
  '''
  if gender==0:
    vMassWeights = [6.68,42.57,2.55,1.38,0.56,2.55,1.38,0.56,
                    14.78,4.81,1.29,14.78,4.81,1.29]
  else:
    vMassWeights = [6.94,43.46,2.71,1.62,0.61,2.71,1.62,0.61,
                    14.16,4.33,1.37,14.16,4.33,1.37]

  CoM = np.zeros(kp.shape[:-2]+(3,))

  CoM += (kp[...,10,:] + kp[...,12,:])/2 * vMassWeights[0] # head+neck
  CoM += kp[...,9,:] * vMassWeights[1] # trunk
  CoM += (kp[...,13,:] + kp[...,14,:])/2 * vMassWeights[2] # upperarm_l
  CoM += (kp[...,14,:] + kp[...,15,:])/2 * vMassWeights[3] # forearm_l
  CoM += kp[...,15,:] * vMassWeights[4] # hand_l
  CoM += (kp[...,16,:] + kp[...,17,:])/2 * vMassWeights[5] # upperarm_r
  CoM += (kp[...,17,:] + kp[...,18,:])/2 * vMassWeights[6] # forearm_r
  CoM += kp[...,18,:] * vMassWeights[7] # hand_r
  CoM += (kp[...,5,:] + kp[...,6,:])/2 * vMassWeights[8] # thigh_l
  CoM += (kp[...,6,:] + kp[...,7,:])/2 * vMassWeights[9] # shank_l
  CoM += (kp[...,7,:] + kp[...,8,:]) * vMassWeights[10] # foot_l
  CoM += (kp[...,1,:] + kp[...,2,:])/2 * vMassWeights[11] # thigh_l
  CoM += (kp[...,2,:] + kp[...,3,:])/2 * vMassWeights[12] # shank_l
  CoM += (kp[...,3,:] + kp[...,4,:]) * vMassWeights[13] # foot_l

  CoM = CoM/100

  return CoM


def contact2phase(contacts):
  stepPhase = []
  phaseON = []
  phaseOFF = []
  l = 0 # left
  for i in range(1,len(contacts)):
    if contacts[i] == 0 and contacts[i-1] == 1:
      phaseON = [l,i]
      l = i
    elif contacts[i] > 0 and contacts[i-1] == 0:
      phaseOFF = [l,i]
      l = i
      stepPhase.append([phaseON, phaseOFF])
  return stepPhase


def calcPosVer(y_0, y_N, N):
  posVertical = []
  for i in range(N):
    y_i = (i/N) * y_N + (N-i) / N * y_0 + 0.5 * G * dt**2 * i*(N-i)
    posVertical.append(y_i)
  return posVertical


# Outputs: Recalulated global poses;
#          recalculated center of mass;
#          step number (landing--0, last step--1, etc.)
# Prerequest: First two frames are step-on-ground
def calcTraj(pose3D, phases, CoMs):
  pre = 0
  cur = 0
  dv = 0
  contJnt = [8,4] #[6, 3] # contact joint

  # Initializing first frame
  pose_global = np.zeros_like(pose3D)
  CoM_global = np.zeros_like(CoMs)
  if pose3D[0,contJnt[1],2]+pose3D[1,contJnt[1],2] >= pose3D[0,contJnt[0],2]+pose3D[1,contJnt[0],2]: # left foot on
    flagFt = 0
    offset = pose3D[0,contJnt[0],:]
  else: # right foot on
    flagFt = 1
    offset = pose3D[0,contJnt[1],:]
  pose_global[0,...] = pose3D[0,...] - offset
  CoM_global[0,...] = CoMs[0,...] - offset

  # Aligning horizontal positions:
  for i,ph in enumerate(phases):
    # ph[0] is ON phase, ph[1] is OFF phase

    pose_global[ph[0][0],:,2] = pose_global[ph[0][0],:,2] - np.min(pose_global[ph[0][0],:,2])
    # ON Phase:
    #for j in range(ph[0][0]+1,ph[0][1]):
    for j in range(ph[0][0]+1,ph[0][1]+1):
      #pose_global[j-1,...] = pose3D[j-1,...]
      #pose_global[j-1,:,2] = pose_global[j-1,:,2] - np.min(pose3D[j-1,:,2])
      offset = pose3D[j,contJnt[flagFt],:] - pose_global[j-1,contJnt[flagFt],:]
      pose_global[j,...] = pose3D[j,...] - offset
      CoM_global[j,...] = CoMs[j,...] - offset

    #dv = (CoM_global[ph[0][1]-1,...] - CoM_global[ph[0][0],...])/(ph[0][1]-1-ph[0][0])
    dv = (CoM_global[ph[0][1],...] - CoM_global[ph[0][0],...])/(ph[0][1]-ph[0][0])
    dv_Hor = np.array([dv[0], dv[1], 0.0])
    # dv_Ver = np.array([0.0, 0.0, dv[2]])
    print(np.linalg.norm(dv_Hor))

    # OFF Phase
    #for idx,j in enumerate(list(range(ph[1][0],ph[1][1]+1))):
    for idx,j in enumerate(list(range(ph[1][0]+1,ph[1][1]+1))):
      CoM_global[j,...] = CoM_global[j-1,...] + dv_Hor
      offset_Hor = CoMs[j,...] - CoM_global[j,...]
      pose_global[j,...] = pose3D[j,...] - offset_Hor
      # offset_Ver = dv_Ver * dt - 0.5 * G * dt**2
      # dv_Ver = dv_Ver - np.array([0.0, 0.0, G * dt])
      # print(dv_Ver)

    flagFt = 1 - flagFt

  #print(pose_global)
  #print(CoM_global)
  """for i in range(pose_global.shape[0]):
    print(pose_global[i,:,2])"""

  # Aligning vertical positions:
  for i,ph in enumerate(phases):
    posVertical = calcPosVer(CoM_global[ph[1][0],2], CoM_global[ph[1][1],2], ph[1][1] - ph[1][0])
    # OFF Phase
    for idx,j in enumerate(list(range(ph[1][0],ph[1][1]))):
      offset_Ver = CoM_global[j,2] - posVertical[idx]
      # print("DEBUG:", CoM_global[j,2], posVertical[idx])
      CoM_global[j,2] = posVertical[idx]
      pose_global[j,:,2] = pose3D[j,:,2] - offset_Ver

  return pose_global, CoM_global


class Features:
  def __init__(self, CoM_global, pose_global, phases, dt):
    self.CoM = CoM_global
    self.pose = pose_global
    self.phase = phases
    self.dt = dt
    self.features = dict()

  def calc_features(self):
    self.features["Overall - effective distance"] = self.calc_Overall_effective_distance()

    time_features = self.calc_Approach_time()
    self.features["Approach - contact time - last"] = time_features[0]
    self.features["Approach - contact time - 2nd"] = time_features[1]
    self.features["Approach - contact time - 3rd"] = time_features[2]
    self.features["Approach - flight time - last"] = time_features[3]
    self.features["Approach - flight time - 2nd"] = time_features[4]
    self.features["Approach - flight time - 3rd"] = time_features[5]
    lengths = self.calc_Approach_length()
    self.features["Approach - step length - last"] = lengths[0]
    self.features["Approach - step length - 2nd"] = lengths[1]
    self.features["Approach - step length - 3rd"] = lengths[2]
    velocities = self.calc_Approach_velocity()
    self.features["Approach - velocity - last"] = velocities[0]
    self.features["Approach - velocity - 2nd"] = velocities[1]
    self.features["Approach - velocity - 3rd"] = velocities[2]

    velocity_TO = self.calc_Takeoff_velocity()
    self.features["TakeOff - horizontal velocity"] = velocity_TO[0]
    self.features["TakeOff - vertical velocity"] = velocity_TO[1]
    self.features["TakeOff - horizontal velocity loss"] = velocity_TO[2]
    self.features["TakeOff - velocity"] = velocity_TO[3]
    self.features["TakeOff - angle"] = velocity_TO[4]

    angle_body = self.calc_Takeoff_body_angles()
    self.features["TakeOff - body inclination at TD"] = angle_body[1]
    self.features["TakeOff - body inclination at TO"] = angle_body[0]

    angle_trunk = self.calc_Takeoff_trunk_angles()
    self.features["TakeOff - trunk inclination at TD"] = angle_trunk[1]
    self.features["TakeOff - trunk inclination at TO"] = angle_trunk[0]

    thigh_features = self.calc_Takeoff_thigh_angles()
    self.features["TakeOff - leading thigh angle at TO"] = thigh_features[0]
    self.features["TakeOff - leading thigh angular velocity"] = thigh_features[1]

    knee_features = self.calc_Takeoff_knee_angles()
    self.features["TakeOff - knee angle at TD"] = knee_features[0]
    self.features["TakeOff - knee angle (minimum)"] = knee_features[1]
    self.features["TakeOff - knee angle motion range"] = knee_features[2]
    self.features["TakeOff - knee angle angular velocity"] = knee_features[3]

    self.features["TakeOff - CoM lowering"] = self.calc_Takeoff_CMlowering()

    landing_features = self.calc_Landing_features()
    self.features["Landing - knee angle"] = landing_features[0]
    self.features["Landing - trunk angle"] = landing_features[1]
    self.features["Landing - landing distance"] = landing_features[2]

  # Functions to calculate features
  # Overall performance
  def calc_dist_two_poses(self, pose1, pose2):
    start_jnt = np.argmin(pose1[:,2])
    end_jnt = np.argmin(pose2[:,2])
    dist = np.linalg.norm(pose2[end_jnt,:2] - pose1[start_jnt,:2])
    return dist

  def calc_Overall_effective_distance(self):
    start_idx = self.phase[-1][1][0] #- 1
    end_idx = self.phase[-1][1][1]
    pose_start = self.pose[start_idx]
    pose_end = self.pose[end_idx]
    dist = self.calc_dist_two_poses(pose_start, pose_end)
    return dist

  # Approach phase
  def calc_Approach_time(self):
    contact_time_1 = (self.phase[-2][0][1] - self.phase[-2][0][0]) * dt
    fligt_time_1 = (self.phase[-2][1][1] - self.phase[-2][1][0]) * dt
    contact_time_2 = (self.phase[-3][0][1] - self.phase[-3][0][0]) * dt
    fligt_time_2 = (self.phase[-3][1][1] - self.phase[-3][1][0]) * dt
    contact_time_3 = (self.phase[-4][0][1] - self.phase[-4][0][0]) * dt
    fligt_time_3 = (self.phase[-4][1][1] - self.phase[-4][1][0]) * dt
    # step time is the sum of contact time and flight time, thus ignore here
    return contact_time_1, contact_time_2, contact_time_3, \
           fligt_time_1, fligt_time_2, fligt_time_3

  def calc_Approach_length(self):
    lengths = []

    for i in range(3):
      st_idx = self.phase[-i-2][1][0] - 1
      ed_idx = self.phase[-i-2][1][1]
      pose_st = self.pose[st_idx]
      pose_ed = self.pose[ed_idx]
      lengths.append(self.calc_dist_two_poses(pose_st, pose_ed))

    return lengths

  def calc_Approach_velocity(self):
    velocities = []
    # Note 1: Frames starting when foot is ON the ground to in the air
    # Note 2: Velocity is the magnitude of full velocity (not horizontal component)
    #         The detailed definition is not mentioned in the report
    for i in range(3):
      st_idx = self.phase[-i-2][0][0]
      ed_idx = self.phase[-i-2][1][1] - 1
      v = np.linalg.norm(self.CoM[ed_idx] - self.CoM[st_idx]) / (ed_idx - st_idx) / self.dt
      velocities.append(v)

    return velocities

  # Take-off phase
  def calc_Takeoff_velocity(self):
    # Take 3 instant positions to calculate the velocity: (-1, 0, +1)
    v_TO = (self.CoM[self.phase[-1][0][1]] - self.CoM[self.phase[-1][0][1]-2]) / 2 / self.dt
    v_TO_Hor = np.linalg.norm(v_TO[:2])
    v_TO_Ver = v_TO[2]
    angle_TO = math.degrees(np.arctan(v_TO_Ver/v_TO_Hor)) # / 2 / 3.14159 * 360.0
    v_TO_Mag = np.linalg.norm(v_TO)
    v_TD = (self.CoM[self.phase[-1][0][0]+1] - self.CoM[self.phase[-1][0][0]-1]) / 2 / self.dt
    v_TD_Hor = np.linalg.norm(v_TD[:2])
    v_Loss_Hor = v_TO_Hor - v_TD_Hor
    # print(v_TO_Hor, v_TO_Ver, v_Loss_Hor, v_TO_Mag, angle_TO, v_TO, v_TD)

    return v_TO_Hor, v_TO_Ver, v_Loss_Hor, v_TO_Mag, angle_TO


  def calc_Takeoff_body_angles(self):
    TO_idx = self.phase[-1][0][1] - 1
    TD_idx = self.phase[-1][0][0]
    vec_dir = self.CoM[TO_idx] - self.CoM[TD_idx] # Vector for determine sign
    vec_dir[2] = 0
    vec_norm = np.array([0.0, 0.0, 1.0])

    TO_jnt = np.argmin(self.pose[TO_idx][:,2])
    vec_TO = self.CoM[TO_idx] - self.pose[TO_idx][TO_jnt,:]
    angle_TO = math.degrees(np.arccos(np.dot(vec_TO, vec_norm) / np.linalg.norm(vec_TO)))
    if np.dot(vec_TO, vec_dir) < 0:
      angle_TO = -angle_TO

    TD_jnt = np.argmin(self.pose[TD_idx][:,2])
    vec_TD = self.CoM[TD_idx] - self.pose[TD_idx][TD_jnt,:]
    angle_TD = math.degrees(np.arccos(np.dot(vec_TD, vec_norm) / np.linalg.norm(vec_TD)))
    if np.dot(vec_TD, vec_dir) < 0:
      angle_TD = -angle_TD

    return angle_TO, angle_TD


  def calc_Takeoff_trunk_angles(self):
    TO_idx = self.phase[-1][0][1] - 1
    TD_idx = self.phase[-1][0][0]
    vec_dir = self.CoM[TO_idx] - self.CoM[TD_idx] # Vector for determine sign
    vec_dir[2] = 0
    vec_norm = np.array([0.0, 0.0, 1.0])

    # Trunk is defined as hip-joint to upper-spine-joint
    vec_TO = self.pose[TO_idx][10,:] - self.pose[TO_idx][0,:]
    angle_TO = math.degrees(np.arccos(np.dot(vec_TO, vec_norm) / np.linalg.norm(vec_TO)))
    if np.dot(vec_TO, vec_dir) < 0:
      angle_TO = -angle_TO

    vec_TD = self.pose[TD_idx][10,:] - self.pose[TD_idx][0,:]
    angle_TD = math.degrees(np.arccos(np.dot(vec_TD, vec_norm) / np.linalg.norm(vec_TD)))
    if np.dot(vec_TD, vec_dir) < 0:
      angle_TD = -angle_TD

    return angle_TO, angle_TD

  def calc_Takeoff_thigh_angles(self):
    TO_idx = self.phase[-1][0][1] - 1
    TD_idx = self.phase[-1][0][0]
    vec_norm = np.array([0.0, 0.0, 1.0])

    if self.pose[TO_idx][3,2] > self.pose[TO_idx][7,2]:
      leg_jnts = [1, 2, 3] # Pelvis, knee, ankle
    else:
      leg_jnts = [5, 6, 7]

    # Thigh angle and angular velocity
    vec_thigh_TO = self.pose[TO_idx][leg_jnts[1],:] - self.pose[TO_idx][leg_jnts[0],:]
    angle_thigh_TO = 90.0 - math.degrees(np.arccos(np.dot(vec_thigh_TO, vec_norm) / np.linalg.norm(vec_thigh_TO)))

    vec_thigh_TD = self.pose[TD_idx][leg_jnts[1],:] - self.pose[TD_idx][leg_jnts[0],:]
    angle_thigh_TD = 90.0 - math.degrees(np.arccos(np.dot(vec_thigh_TD, vec_norm) / np.linalg.norm(vec_thigh_TD)))

    angular_velocity_thigh = (angle_thigh_TO - angle_thigh_TD) / (TO_idx - TD_idx) / self.dt

    return angle_thigh_TO, angular_velocity_thigh

  def calc_Takeoff_knee_angles(self):
    # The knee of the supporting leg
    TO_idx = self.phase[-1][0][1] - 1
    TD_idx = self.phase[-1][0][0]
    vec_norm = np.array([0.0, 0.0, 1.0])

    if self.pose[TO_idx][3,2] < self.pose[TO_idx][7,2]:  # Opposite to swing leg
      leg_jnts = [1, 2, 3] # Pelvis, knee, ankle
    else:
      leg_jnts = [5, 6, 7]

    angles_knee = []
    for idx in range(TD_idx, TO_idx + 1):
      vec_thigh = self.pose[idx][leg_jnts[0],:] - self.pose[idx][leg_jnts[1],:]
      vec_calf = self.pose[idx][leg_jnts[2],:] - self.pose[idx][leg_jnts[1],:]
      angle = math.degrees(np.arccos(np.dot(vec_thigh, vec_calf) / np.linalg.norm(vec_thigh) \
                                           / np.linalg.norm(vec_calf)))
      angles_knee.append(angle)

    min_idx = np.argmin(angles_knee)

    angle_knee_TD = angles_knee[0]
    angle_knee_min = angles_knee[min_idx]
    angle_range_knee = angle_knee_TD - angle_knee_min
    angular_velocity_knee = - angle_range_knee / min_idx / self.dt

    return angle_knee_TD, angle_knee_min, angle_range_knee, angular_velocity_knee

  def calc_Takeoff_CMlowering(self):
    # The CoM height reduction from the take-off of the last step to lowest on the board
    TO_last_idx = self.phase[-2][0][1] - 1
    TD_idx = self.phase[-1][0][0]
    TO_idx = self.phase[-1][0][1] - 1
    CoM_origin = self.CoM[TO_last_idx][2]
    CoM_lowest = np.min(self.CoM[TD_idx:TO_idx+1][:,2])
    CoM_lowering = (CoM_origin - CoM_lowest)*100.0

    return CoM_lowering

  # Landing phase
  def calc_Landing_features(self):
    idx = self.phase[-1][1][1]
    pose_landing = self.pose[idx]
    CoM_landing = self.CoM[idx]

    if self.pose[idx][3,2] < self.pose[idx][7,2]:  # Opposite to swing leg
      leg_jnts = [1, 2, 3] # Pelvis, knee, ankle
    else:
      leg_jnts = [5, 6, 7]

    # angle hip --> Definition not clear
    # angle knee
    vec_thigh = self.pose[idx][leg_jnts[0],:] - self.pose[idx][leg_jnts[1],:]
    vec_calf = self.pose[idx][leg_jnts[2],:] - self.pose[idx][leg_jnts[1],:]
    angle_knee = math.degrees(np.arccos(np.dot(vec_thigh, vec_calf) / np.linalg.norm(vec_thigh) \
                                   / np.linalg.norm(vec_calf)))
    # angle trunk
    vec_dir = self.CoM[idx] - self.CoM[idx-1] # Vector for determine sign
    vec_dir[2] = 0
    vec_norm = np.array([0.0, 0.0, 1.0])

    vec_Trunk = pose_landing[10,:] - pose_landing[0,:]
    angle_Trunk = math.degrees(np.arccos(np.dot(vec_Trunk, vec_norm) / np.linalg.norm(vec_Trunk)))
    if np.dot(vec_Trunk, vec_dir) < 0:
      angle_Trunk = -angle_Trunk

    # landing distance (CoM to Landing contact)
    dist_landing = np.linalg.norm(CoM_landing - pose_landing[leg_jnts[2]])

    return angle_knee, angle_Trunk, dist_landing


def eval_feature_estimation(features, gt):
    # print(gt)
    Excludes = ['Approach - contact time - last', \
                'Approach - contact time - 2nd', \
                'Approach - contact time - 3rd', \
                'Approach - flight time - last', \
                'Approach - flight time - 2nd', \
                'Approach - flight time - 3rd', \
                'Approach - step length - last', \
                'Approach - step length - 2nd', \
                'Approach - velocity - last', \
                'Approach - velocity - 2nd']
    i = 0
    for k in features.keys():
        if k not in Excludes:
            if gt[i]=='-':
                continue
            else:
                gtf = float(gt[i])
                print(k,':',np.abs((features[k] - gtf)/gtf) * 100, '%')
                i = i + 1
