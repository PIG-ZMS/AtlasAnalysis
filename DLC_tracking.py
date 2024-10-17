# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:20:45 2024

@author: s2764793
"""
import os
import pandas as pd
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from datetime import datetime

parameter = {
    'tracking_split_tag':'cam',
    'tracking_file_sufix': '.csv',
    'tracking_file_tag': 'DLC',
    'DLC_folder_tag': 'DLC_tracking',
    'well_coord': [[462,211],[314,372]],
    'bridge_coord': [[37,216],[100,258]],
    'CB_centre': [307,215],
    'CB_radius': 209,
    'detecting_radius': 15,
    'point_of_no_return': 1.5,
    'CamFs':30,
    'sig_level':0.9
    }

parent_folder = None
frame_rate = None
class frame:
    def __init__(self,head,shoulder,bottom):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom
        self.well1 = parameter['well_coord'][0]
        self.well2 = parameter['well_coord'][1]
        self.r = parameter['detecting_radius']
        #Not that the origin is at the upper left corner
        self.ang = -math.degrees(Ang(self.shoulder,self.head))
    
    def IsCloseToWell(self):
        if Dis(self.well1,self.well2)<=self.r*2:
            print('Error,the two wells are too close!')
            return
        if Dis(self.head,self.well1)<=self.r:
            self.well = self.well1
            self.well_tag = 1
            # print('Close to well 1')
            if self.IsInward():
                return True
        if Dis(self.head,self.well2)<=self.r:
            self.well = self.well2
            self.well_tag = 2
            # print('Close to well 2')
            if self.IsInward():
                return True
        return False
    
    def IsInward(self):
        if Dis(self.head,self.well)<=Dis(self.shoulder,self.well):
            return True
        else:
            return False
    
    def IsOnBridge(self):
        x1 = min(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])
        x2 = max(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])
        y1 = min(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])
        y2 = max(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])
        
        #This mouse is on bridge if at least two body parts are in the area
        bodyparts_onbridge = 0
        if (self.head[0]>=x1 and self.head[0]<=x2) and (self.head[1]>=y1 and self.head[1]<=y2):
            bodyparts_onbridge+=1
        if (self.shoulder[0]>=x1 and self.shoulder[0]<=x2) and (self.shoulder[1]>=y1 and self.shoulder[1]<=y2):
            bodyparts_onbridge+=1
        if (self.bottom[0]>=x1 and self.bottom[0]<=x2) and (self.bottom[1]>=y1 and self.bottom[1]<=y2):
            bodyparts_onbridge+=1
        if bodyparts_onbridge>=2:
            return True
        else:
            return False
    
    #This determined the head direction of mouse (whether it is entering or leaving CB)
    def BridgeDir(self):
        if (self.head[0]>=self.shoulder[0]) and (self.shoulder[0]>=self.bottom[0]):
            #This imply the mouse is heading toward CB (entering)
            return 1
        elif (self.head[0]<=self.shoulder[0]) and (self.shoulder[0]<=self.bottom[0]):
            #This imply the mouse is heading toward SB (leaving)
            return 2
        else:
            return 0
    
    # def IsValid(self):
    def IsInCB (self):
        bodyparts_CB = 0
        r = parameter['CB_radius']
        if Dis(self.head,parameter['CB_centre'])<=r:
            bodyparts_CB+=1
        if Dis(self.shoulder,parameter['CB_centre'])<=r:
            bodyparts_CB+=1
        if Dis(self.bottom,parameter['CB_centre'])<=r:
            bodyparts_CB+=1
        if bodyparts_CB>=2:
            return True
        else:
            return False
        
    
class trace:
    def __init__(self,file,ID,day):
        self.file = file
        self.ID = ID
        self.day = day
        self.frames = []
        self.BoarderForce = {
            'Time':[],
            'Event':[],
            'Location':[]
            }
        for i in range (file.shape[0]):
            if file.iloc[i,0] == 'bodyparts':
                self.bdp_index = i
            if file.iloc[i,0] == 'coords':
                self.xy_index = i
        for i in range (file.shape[0]):
            if not file.iloc[i,0].isdigit():
                continue
            for j in range (file.shape[1]):
                if (file.iloc[self.bdp_index,j] == 'head'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        head_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        head_y = float(file.iloc[i,j])
                if (file.iloc[self.bdp_index,j] == 'shoulder'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        shoulder_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        shoulder_y = float(file.iloc[i,j])
                if (file.iloc[self.bdp_index,j] == 'bottom'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        bottom_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        bottom_y = float(file.iloc[i,j])
            single_frame = frame([head_x,head_y],[shoulder_x,shoulder_y],[bottom_x,bottom_y])
            self.frames.append(single_frame)
        
        self.SaveFile()
        self.Marking()
        self.PlotTraceMap(max(self.start_frame,0),min(self.end_frame,len(self.frames)))
        
        
        x = min(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])
        y = min(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])
        l = max(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])-x
        h = max(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])-y
        square = Rectangle([x,y], l, h, edgecolor='black', facecolor='none', linewidth=1)
        circle1 = plt.Circle(parameter['well_coord'][0], parameter['detecting_radius'], color='r', fill=True)
        circle2 = plt.Circle(parameter['well_coord'][1], parameter['detecting_radius'], color='b', fill=True)
        circle3 = plt.Circle(parameter['CB_centre'],parameter['CB_radius'], color='black',fill=False)
        # Add the circles to the current plot
        plt.gca().add_patch(square)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        plt.xlim(0, 640)
        plt.ylim(0, 480)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        
        name = 'Day'+str(self.day)+'-'+str(self.ID)+'Tracking_map.png'
        output_path = os.path.join(output_folder,'Tracking map')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path,name))
        plt.close()
        
    def Marking (self,):
        frame_stamp = 0
        current_state = 'Outside'
        CB = False
        has_entered_CB_before = False
        current_well = None
        self.Bridge_detection = False
        self.enter_CB = -1
        self.leave_CB = -1
        self.leave_SB = -1
        self.start_frame = -1
        self.end_frame = 999999999999
        
        for i in range (len(self.frames)):
            frame = self.frames[i]
            if (not CB) and frame.IsInCB() and self.CBSig(i):
                CB = True
                if not has_entered_CB_before:
                    self.start_frame = i
                has_entered_CB_before = True
                self.enter_CB = i/frame_rate
                self.BoarderForce['Time'].append(i/frame_rate)
                self.BoarderForce['Event'].append('Enter CB')
                self.BoarderForce['Location'].append('-')
                print('In CB at:'+str(i/frame_rate))
            if CB and not frame.IsInCB() and self.CBSig(i):
                CB = False
                self.leave_CB = i/frame_rate
                self.end_frame = i
                self.BoarderForce['Time'].append(i/frame_rate)
                self.BoarderForce['Event'].append('Leave CB')
                self.BoarderForce['Location'].append('-')
                print('Leave CB at:'+str(i/frame_rate))
        
            #This refers that the mouse may trying to reach the well
            if (frame.IsCloseToWell() and current_state == 'Outside'):
                if self.BoarderPass(i,True):
                    current_state = 'Inside'
                    self.BoarderForce['Time'].append(i/frame_rate)
                    self.BoarderForce['Event'].append('Approaching Well')
                    self.BoarderForce['Location'].append('Well'+str(frame.well_tag))
                    current_well = frame.well_tag
            if (not frame.IsCloseToWell() and current_state == 'Inside'):
                if self.BoarderPass(i,False):
                    current_state = 'Outside'
                    self.BoarderForce['Time'].append(i/frame_rate)
                    self.BoarderForce['Event'].append('Leaving Well')
                    self.BoarderForce['Location'].append('Well'+str(current_well))
                    
            if (frame.IsOnBridge() and self.BridgeSig(i) and not has_entered_CB_before and self.leave_SB==-1):
                self.Bridge_detection = True
                self.leave_SB = i/frame_rate
                self.BoarderForce['Time'].append(i/frame_rate)
                self.BoarderForce['Event'].append('Leave SB')
                self.BoarderForce['Location'].append('-')
                print('Leaving SB at:'+str(i/frame_rate))
            
            if (not frame.IsOnBridge() and not self.BridgeSig(i) and not frame.IsInCB()):
                self.frames[i] = self.frames[i-1]
        
    def CBSig(self,index):
        exp = self.frames[index].IsInCB()
        frame_of_no_return = round(parameter['point_of_no_return']*frame_rate)
        legit = 0
        outlaw = 0
        for i in self.frames[index:min(len(self.frames),index+frame_of_no_return)]:
            if (i.IsInCB() == exp):
                legit+=1
            else:
                outlaw+=1
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            return True
        else:
            return False
    
    def BridgeSig (self,index):
        expectation = self.frames[index].IsOnBridge()
        #The mouse is identified as on bridge if its remain on the bridge in the following 0.5s
        frame_of_no_return = round(0.3*frame_rate)
        legit = 0
        outlaw = 0
        for i in self.frames[index:min(len(self.frames),index+frame_of_no_return)]:
            if (i.IsOnBridge() == expectation):
                legit+=1
            else:
                outlaw+=1
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            
            return True
        else:
            return False
    

    
    def BoarderPass (self,index,expectation):
        
        frame_of_no_return = round(parameter['point_of_no_return']*frame_rate)
        legit = 0
        outlaw = 0
        for i in self.frames[index:min(len(self.frames),index+frame_of_no_return)]:
            if (i.IsCloseToWell() == expectation):
                legit+=1
            else:
                outlaw+=1
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            return True
        else:
            return False
    
    def ObtainShoulderCoord(self,f1,f2):
        x_values = [frame.shoulder[0] for frame in self.frames[f1:f2]]
        y_values = [frame.shoulder[1] for frame in self.frames[f1:f2]]
        return x_values,y_values        
        
    def SaveFile (self):
        data = {
            'head_x':[],
            'head_y':[],
            'shoulder_x':[],
            'shoulder_y':[],
            'bottom_x':[],
            'bottom_y':[],
            'angle':[],
            'speed':[],
            'isinCB':[],
            'isclosetowell':[]
            }
        for index,i in enumerate(self.frames):
            if index != 0:
                data['speed'].append(Dis(self.frames[index-1].shoulder,self.frames[index].shoulder)*frame_rate)
            else:
                data['speed'].append(0)
            
            
            
            data['head_x'].append(i.head[0])
            data['head_y'].append(i.head[1])
            data['shoulder_x'].append(i.shoulder[0])
            data['shoulder_y'].append(i.shoulder[1])
            data['bottom_x'].append(i.bottom[0])
            data['bottom_y'].append(i.bottom[1])
            data['angle'].append(i.ang)
            data['isinCB'].append(i.IsInCB())
            data['isclosetowell'].append(i.IsCloseToWell())
                
        self.df = pd.DataFrame(data)
     
    def PlotTraceMap (self,start_frame,end_frame):
        x,y = self.ObtainShoulderCoord(start_frame,end_frame)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with a colormap that gradually changes along the line segments
        lc = LineCollection(segments, cmap='viridis', linewidth=2)
        lc.set_array(np.linspace(round(start_frame/parameter['CamFs']), round(end_frame/parameter['CamFs']), len(x) - 1))  # Set color based on the index
        
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Time')
        lc.set_clim(round(start_frame/parameter['CamFs']), round(end_frame/parameter['CamFs']))
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trace with Gradual Color Change')
        
class dayx:
    def __init__(self,folder,day,op_path):
        self.day = day
        self.trails = []
        output_path = os.path.join(op_path,'Day'+str(day))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(folder):
            if filename.endswith(parameter['tracking_file_sufix']) and parameter['tracking_file_tag'] in filename:
                file_path = os.path.join(folder,filename)
                ID = int(re.findall(r'\d+', filename.split(parameter['tracking_split_tag'])[1])[0])
                file = pd.read_csv(file_path)
                print('Day'+str(self.day)+' Trial'+str(ID))
                t = trace(file,ID,self.day)
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in t.BoarderForce.items()]))
                df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_tracking.csv'))
                t.df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_frames.csv'))
                self.trails.append(t)  

class mice:
    def __init__(self,parent_folder,mouse_ID):
        self.mice_days = []
        output_path = os.path.join(parent_folder,'DLC_output')
        global output_folder
        output_folder = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('Now reading:'+str(mouse_ID))
        for filename in os.listdir(parent_folder):
            if parameter['DLC_folder_tag'] in filename:
                print(filename)
                if 'day' in filename:
                    day = int(re.findall(r'\d+', filename.split('day')[1])[0])
                    folder = os.path.join(parent_folder,filename)
                elif 'Day' in filename:
                    day = int(re.findall(r'\d+', filename.split('Day')[1])[0])
                    folder = os.path.join(parent_folder,filename)
                elif 'Probe' in filename or 'probe' in filename:
                    day = -1
                    folder = os.path.join(parent_folder,filename)
                else:
                    continue
                self.mice_days.append(dayx(folder,day,output_path))
                # self.Neo_Cold(parent_folder)
     
    def Neo_Cold(self,folder):
        cold = {
            'Day':[],
            'Trail_ID':[],
            'Time_of_leaving_SB':[]
            }
        path = os.path.join(folder,'Neo_Cold.csv')
        print(path)
        for day in self.mice_days:
            for trail in day.trails:
                cold['Day'].append(day.day)
                cold['Trail_ID'].append(trail.ID)
                cold['Time_of_leaving_SB'].append(trail.leave_SB)
        df = pd.DataFrame(cold)
        df.to_csv(path)

def Dis (x,y):
    return np.sqrt((y[0]-x[0])**2+(y[1]-x[1])**2)

def Ang (x,y):
    return math.atan2((y[1]-x[1]),(y[0]-x[0]))

def ObtainFrameRate(parent_folder):
    for file in os.listdir(parent_folder):
        if 'Bonsai' in file:
            path = os.path.join(parent_folder,file)
            for name in os.listdir(path):
                if 'sync' in name:
                    df = pd.read_csv(os.path.join(path,name))
                    start_timestamp = df['Timestamp'].iloc[0]
                    end_timestamp = df['Timestamp'].iloc[-1]
                    start_time = datetime.fromisoformat(start_timestamp)
                    end_time = datetime.fromisoformat(end_timestamp)
                    duration = (end_time - start_time).total_seconds()
                    global frame_rate
                    frame_rate = df.shape[0] / duration
                    
global output_folder
# folder = 'D:/Photometry/test_tracking/1786535/'
folder = 'D:/Photometry/test_tracking/Group B/1756074/'
ObtainFrameRate(folder)
print('Frame is:'+str(frame_rate))
a = mice(folder,'1756074')

                
                