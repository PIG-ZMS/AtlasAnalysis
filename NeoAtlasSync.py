# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:07:53 2024

@author: mingshuai zhu
"""

import os
import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
import photometry_functions as pf
import seaborn as sns
from datetime import datetime
from scipy.stats import sem,t
import seaborn
import random
import scipy.signal as signal

parameter = {
    'grandparent_folder':'D:/Photometry/Atlas_folder/',
    'DLC_folder_tag': 'DLC_output',
    'Bonsai_folder_tag': 'Bonsai',
    'sync_tag': 'sync',
    'atlas_tag': 'Atlas',
    'atlas_z_filename':'Zscore_trace.csv',
    'atlas_green_filename':'Green_trace.csv',
    'atlas_frame_rate': 840,
    'CamFs' : 30,
    'before_win': 1,
    'after_win': 1,
    'low_pass_filter_frequency': 80
    }

pfw = 0
output_path = None
mouse_name = None
has_SB_recording = False
test = None


        
class DLCframefile:
    def __init__(self,path,name):
        self.day = int(re.findall(r'\d+', name.split('-')[0])[0])
        self.ID = int(re.findall(r'\d+', name.split('-')[1])[0])
        self.df = pd.read_csv(path)


class DLCtrackingfile:
    def __init__(self,path,name):
        self.day = int(re.findall(r'\d+', name.split('-')[0])[0])
        self.ID = int(re.findall(r'\d+', name.split('-')[1])[0])
        self.df = pd.read_csv(path)
    
class sync_file:
    def __init__(self,path,day,ID):
        self.path = path
        self.day = day
        self.ID = ID
        self.df = pd.read_csv(path)
        self.CalculateFrameRate()
        self.ObtainFrameShift()
        
    def CalculateFrameRate(self):
        start_timestamp = self.df['Timestamp'].iloc[0]
        end_timestamp = self.df['Timestamp'].iloc[-1]
        start_time = datetime.fromisoformat(start_timestamp)
        end_time = datetime.fromisoformat(end_timestamp)
        duration = (end_time - start_time).total_seconds()
        self.frame_rate = self.df.shape[0] / duration
        
    def ObtainFrameShift (self):
        for i in range (self.df.shape[0]):
            if np.isnan(self.df['Value.X'].iloc[i]) and np.isnan(self.df['Value.Y'].iloc[i]):
                self.frame_shift = i
                return
    
    
class z_file:
    def __init__(self,path,day,ID):
        self.path = path
        self.day = day
        self.ID = ID
        self.df = pd.read_csv(path,header=None)
        self.signal = self.df[0]
        self.filtered_signal = LowPassFilter(self.signal)
        
    def ObtainEventSignal(self,time):
        before_frame = round((time-parameter['before_win'])*parameter['atlas_frame_rate'])
        after_frame = round((time+parameter['after_win'])*parameter['atlas_frame_rate'])
        #This imply that the recording does not fully cover the event
        if (before_frame<0 or after_frame>=self.df.shape[0]):
            return -1
        
        return np.array(self.filtered_signal[before_frame:after_frame])

class green_file:
    def __init__(self,path,day,ID):
        self.path = path
        self.day = day
        self.ID = ID
        self.df = pd.read_csv(path,header=None)
        self.signal = self.df[0]
        self.filtered_signal = LowPassFilter(self.signal)
    
        


class singletrail:
    def __init__(self,frame,day):
        self.frame = frame
        self.day = day
        self.ID = self.frame.ID
        self.IsImportant = False
    
    def ItIsImportant(self):
        self.IsImportant = True
    
    def AddTrackingFile (self,tracking):
        self.tracking = tracking
    
    def AddSyncFile (self,sync):
        self.sync = sync
        
    def AddAtlasFile(self,folder):
        for file in os.listdir(folder):
            if file == parameter['atlas_z_filename']:
                path = os.path.join(folder,file)
                self.z = z_file(path,self.day,self.ID)
            if file == parameter['atlas_green_filename']:
                path = os.path.join(folder,file)
                self.green = green_file(path,self.day,self.ID)
    
    def EventAnalysis (self):
        timeshift_video = self.sync.frame_shift/self.sync.frame_rate
        self.time_dif = timeshift_video
        self.reward_collection = []
        self.Leaving_well = []
        self.leaving_SB = None
        for i in range (self.tracking.df.shape[0]):
            if self.tracking.df['Event'].iloc[i] == 'Leave SB':
                self.leaving_SB = (self.tracking.df['Time'].iloc[i])
            if self.tracking.df['Event'].iloc[i] == 'Approaching Well':
                self.reward_collection.append([self.tracking.df['Time'].iloc[i],self.tracking.df['Location'].iloc[i]])
            if self.tracking.df['Event'].iloc[i] == 'Leaving Well':
                self.Leaving_well.append([self.tracking.df['Time'].iloc[i],self.tracking.df['Location'].iloc[i]])
        
        self.LeavingSBSignal()
        self.PlotRewardCollection()
        self.PlotLeavingWell()
        self.ObtainSpeedAng()
        
        if self.IsImportant:
            self.SaveGreenAndSpeed()
        
    
    def LeavingSBSignal (self):
        D = ['Day'+str(self.day)]
        T = ['Trail'+str(self.ID)]
        if self.leaving_SB:
            z = self.z.ObtainEventSignal(self.leaving_SB-self.time_dif)
        else:
            self.leaving_SB_signal = pd.DataFrame()
            return
        if isinstance(z, np.ndarray):
            name = 'Day'+str(self.day)+'-'+str(self.ID)
            fig, ax = plt.subplots()
            fig, ax = PlotEvent(z,ylab = 'z_score', title = name, event = 'Leaving SB')
            path = os.path.join(output_path,'Leaving_SB')
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path,name))
            
        header = pd.MultiIndex.from_arrays([D,T])
        if isinstance(z, np.ndarray):
            self.leaving_SB_signal = pd.DataFrame(np.array(z).T,columns=header)
            self.leaving_SB_signal.columns.names = ['Day', 'Trail']
        else:
            self.leaving_SB_signal = pd.DataFrame()
            
    def PlotRewardCollection (self):
        index = 0
        reward_signals_list = []
        self.AUC_dif_list = []
        D = []
        T = []
        name = 'Day'+str(self.day)+'-'+str(self.ID)
        for i in range(len(self.reward_collection)):
            z = self.z.ObtainEventSignal(self.reward_collection[i][0]-self.time_dif)
            
            if not isinstance(z, np.ndarray):
                continue
            
            reward_signals_list.append(z)
            D.append(f"Day{self.day}")
            T.append(f"Trail{self.ID}")
            fig, ax = plt.subplots()
            fig, ax = PlotEvent(z,ylab = 'z_score', title = name, event = 'Approaching '+self.reward_collection[i][1])
            path = os.path.join(output_path,'Reward_Collection')
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, f"{name}({index})"))
            l = len(z)
            seg_a = z[0:int(l/2)]
            seg_b = z[int(l/2):int(l)]
            area1 = np.trapz(seg_a)
            area2 = np.trapz(seg_b)
            self.AUC_dif_list.append(area1-area2)
            index+=1
            
        header = pd.MultiIndex.from_arrays([D,T])
        if reward_signals_list:
            self.reward_signals = pd.DataFrame(np.array(reward_signals_list).T,columns=header)
            self.reward_signals.columns.names = ['Day', 'Trail']
        else:
            self.reward_signals = pd.DataFrame()
        
            
    def PlotLeavingWell(self):
        index = 0
        leave_signals_list = []
        D = []
        T = []
        name = 'Day'+str(self.day)+'-'+str(self.ID)
        for i in range(len(self.Leaving_well)):
            z = self.z.ObtainEventSignal(self.Leaving_well[i][0]-self.time_dif)
            if not isinstance(z, np.ndarray):
                continue
            leave_signals_list.append(z)
            D.append(f"Day{self.day}")
            T.append(f"Trail{self.ID}")
            fig, ax = plt.subplots()
            fig, ax = PlotEvent(z,ylab = 'z_score', title = name, event = 'Leaving '+self.reward_collection[i][1])
            path = os.path.join(output_path,'Leaving_well')
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, f"{name}({index})"))
            index+=1
            
        header = pd.MultiIndex.from_arrays([D,T])
        if leave_signals_list:
            self.leaving_signals = pd.DataFrame(np.array(leave_signals_list).T,columns=header)
            self.leaving_signals.columns.names = ['Day', 'Trail']
        else:
            self.leaving_signals = pd.DataFrame()
     
    def ObtainSpeedAng(self):
        dic = {
            'z_score':[],
            'speed':[],
            'angle':[]
            }
        
        for i in range (self.frame.df.shape[0]):
            if self.frame.df['isinCB'].iloc[i] and not self.frame.df['isclosetowell'].iloc[i]:
                time = i/self.sync.frame_rate-self.time_dif
                x = round(time*parameter['atlas_frame_rate'])
                if x>=0 and x<len(self.z.signal):
                    z = self.z.signal[x]
                    dic['z_score'].append(z)
                    dic['speed'].append(self.frame.df['speed'].iloc[i])
                    dic['angle'].append(self.frame.df['angle'].iloc[i])    
        
        self.velocity = pd.DataFrame(dic)
    
    def CalculatePowerRatio (self):
        fs = parameter['atlas_frame_rate'] 
        # Calculate the Power Spectral Density using Welch's method
        # Convert z.signal DataFrame to a 1D NumPy array
        signal_data = self.z.signal.to_numpy().flatten()
    
        # Calculate the Power Spectral Density using Welch's method
        frequencies, psd = signal.welch(signal_data, fs, nperseg=256)
        # Define frequency ranges
        low_freq_band = (frequencies <= 10)
        high_freq_band = (frequencies > 10)
        
        # Calculate the total power in each band
        low_power = np.sum(psd[low_freq_band])
        high_power = np.sum(psd[high_freq_band])
        
        # Calculate the high/low frequency power ratio
        self.power_ratio = low_power / high_power
     
    def SaveGreenAndSpeed (self):
        path = os.path.join(output_path,'speed_files')
        path = os.path.join(path,f"Day{self.day}")
        if not os.path.exists(path):
            os.makedirs(path)
        recorded_speed = []
        for i in range (self.z.signal.shape[0]):
            recorded_speed.append(self.frame.df['speed'].iloc[self.sync.frame_shift+round(i/parameter['atlas_frame_rate']*self.sync.frame_rate)])
        
        data = {
            'raw_z_score':self.z.signal,
            'raw_green':self.green.signal,
            'filtered_z_score':self.z.filtered_signal,
            'filtered_green':self.green.filtered_signal,
            'instant_speed':recorded_speed
            }
        name = f"Green&Speed_Day{self.day}-{self.ID}.csv"
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(path,name), index=False)
        
class singleday:
    #first reading in DLC tracking files
    def __init__(self,folder,name):
        self.trails = []
        if 'Probe' in name:
            self.day = -1
        else:
            self.day = int(re.findall(r'\d+', name)[0])
        for file in os.listdir(folder):
            path = os.path.join(folder,file)
            if 'frames' in file:
                print(file)
                frame_file = DLCframefile(path,file)
                trail = singletrail(frame_file,self.day)
                self.trails.append(trail)
                
        self.trails_dic = {trail.ID: trail for trail in self.trails}      
        
        for file in os.listdir(folder):
            path = os.path.join(folder,file)
            if 'tracking' in file:
                tracking_file = DLCtrackingfile(path,file)
                self.trails_dic[tracking_file.ID].AddTrackingFile(tracking_file)
        
    def AddBonsai (self,folder):
        for file in os.listdir(folder):
            if parameter['sync_tag'] in file:
                path = os.path.join(folder,file)
                trail_ID = int(re.findall(r'\d+', file.split(parameter['sync_tag'])[1])[0])
                trail_sync = sync_file(path,self.day,trail_ID)
                self.trails_dic[trail_ID].AddSyncFile(trail_sync)
            if file.endswith('.docx'):
                key_index = re.findall(r'\d+', file)[0]
                self.keynum = []
                
                pre_index = -1
                current_index = 0
                
                #aiming to deal with more than 10 trails
                for i in range (len(key_index)):
                    current_index += int(key_index[i])
                    if current_index > pre_index:
                        pre_index = current_index
                        self.trails_dic[current_index].ItIsImportant()
                        self.keynum.append(current_index)
                        current_index = 0
                    else:
                        current_index *= 10
                
                
    def AddAtlas (self,folder):
        for file in os.listdir(folder):
                path = os.path.join(folder,file)
                index = int(re.findall(r'\d+', file)[-1])
                self.trails_dic[self.keynum[index-1]].AddAtlasFile(path)
                
                
    def Analysis(self):
        self.day_z = []
        
        for trail in self.trails:
            if trail.IsImportant:
                trail.CalculatePowerRatio()
                self.day_z.append(np.array(trail.z.signal))

        # self.day_z = np.concatenate(self.day_z, axis=0)
        
        self.tot_velocity = pd.DataFrame()
        self.tot_reward_signal = pd.DataFrame()
        self.tot_leaving_signal = pd.DataFrame()
        self.tot_leaving_SB = pd.DataFrame()
        for trail in self.trails:    
            if trail.IsImportant:
                trail.EventAnalysis()
                self.tot_reward_signal = pd.concat([self.tot_reward_signal,trail.reward_signals],axis=1)
                self.tot_leaving_signal = pd.concat([self.tot_leaving_signal,trail.leaving_signals],axis=1)
                self.tot_velocity = pd.concat([self.tot_velocity,trail.velocity],axis=0)
                self.tot_leaving_SB = pd.concat([self.tot_leaving_SB,trail.leaving_SB_signal],axis=1)
        fig,ax = PlotEventMean(self.tot_reward_signal,title=f"Day{self.day} mean reward signal",event='reward_collection')
        path = os.path.join(output_path,'Reward_Collection')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} mean"))
        
        fig,ax = PlotEventMean(self.tot_leaving_SB,title=f"Day{self.day} mean leaving signal",event='Leaving well')
        path = os.path.join(output_path,'Leaving_SB')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} mean"))
        
        fig,ax = PlotEventMean(self.tot_leaving_signal,title=f"Day{self.day} mean leaving SB signal",event='Leaving SB')
        path = os.path.join(output_path,'Leaving_well')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} mean"))
        

        
        fig,ax = PlotSpeedHistogram(self.tot_velocity['z_score'], self.tot_velocity['speed'])
        path = os.path.join(output_path,'speed')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} speed"))
        
        fig,ax = PlotSpeedHistogram(self.tot_velocity['z_score'], self.tot_velocity['speed'],title=f"Day{self.day} speed vs z_score")
        path = os.path.join(output_path,'speed')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} speed"))
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax = PlotRadianGraph(self.tot_velocity['z_score'], self.tot_velocity['angle'], ax, label='Original Mean', color='purple')
        #setting up bootstrap analysis
        neo_z_score = []
        for i in range (len(self.tot_velocity['angle'])):
            r = random.randint(0, len(self.tot_velocity['z_score'])-1)
            neo_z_score.append(self.tot_velocity['z_score'].iloc[r])
        ax = PlotRadianGraph(neo_z_score, self.tot_velocity['angle'], ax, label='Bootstrap Mean', color='orange', linestyle='--',title=f"Day{self.day} mean z_score at different head direction")
        ax.legend()
        path = os.path.join(output_path,'angle')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Day{self.day} angle"))
    
class mice:
    def __init__(self,folder,mouse_ID):
        self.photometry_files = []
        self.DLC_pairs = []
        self.frames_files = []
        self.tracking_files = []
        self.mouse_ID = mouse_ID
        self.days = []
        global mouse_name
        mouse_name = self.mouse_ID
        
        for file in os.listdir(folder):
            #finding DLC folder
            if parameter['DLC_folder_tag'] in file:
                print(file)
                DLC_folder = os.path.join(folder,file)
                break
            
        #Reading in DLC files
        for file in os.listdir(DLC_folder):
            if 'Day' in file or 'Probe' in file:
                path = os.path.join(DLC_folder,file)
                day = singleday(path,file)
                self.days.append(day)
                print(f"Day {day.day}")
        day_dic = {x.day: x for x in self.days}
        
        #Finding Bonsai folder
        for file in os.listdir(folder):
            if parameter['Bonsai_folder_tag'] in file:
                print(file)
                Bonsai_folder = os.path.join(folder,file)
                if 'Probe' in file:
                    d = -1
                else:
                    d = int(re.findall(r'\d+',file)[0])
                day_dic[d].AddBonsai(Bonsai_folder)
        
        #Reading in CB folder
        for file in os.listdir(folder):
            if parameter['atlas_tag'] in file:
                print(file)
                atlas_folder = os.path.join(folder,file)
                if 'Probe' in file:
                    d = -1
                else:
                    d = int(re.findall(r'\d+',file)[0])
                day_dic[d].AddAtlas(atlas_folder)
        
        for day in self.days:
            day.Analysis()
            
    def Analysis(self):
        dic = {}
        index = 0
        for day in self.days:
            name = 'Day'+str(day.day)
            dic[name+'z_score'] = day.day_z
                        
        fig,ax = PlotDicMean(dic,ylab='z_score_mean',title='z_score mean with error bar')
        fig.savefig(os.path.join(output_path,'CBSB_comparision'))
        
        self.tot_reward_signal = pd.DataFrame()
        self.tot_leaving_signal = pd.DataFrame()
        self.tot_velocity = pd.DataFrame()
        self.tot_leaving_SB = pd.DataFrame()
        for day in self.days:
            self.tot_reward_signal = pd.concat([self.tot_reward_signal,day.tot_reward_signal],axis=1)
            self.tot_leaving_signal = pd.concat([self.tot_leaving_signal,day.tot_leaving_signal],axis=1)
            self.tot_velocity = pd.concat([self.tot_velocity,day.tot_velocity],axis=0)
            self.tot_leaving_SB = pd.concat([self.tot_leaving_SB,day.tot_leaving_SB],axis=1)
        fig,ax = PlotEventHeatMap(self.tot_reward_signal,title='Reward collection heatmap')
        path = os.path.join(output_path,'Reward_Collection')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'heatmap'))
        
        fig,ax = PlotEventHeatMap(self.tot_leaving_signal,title='leaving well heatmap')
        path = os.path.join(output_path,'Leaving_well')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'heatmap'))
        
        fig,ax = PlotEventHeatMap(self.tot_leaving_SB,title='leaving SB heatmap')
        path = os.path.join(output_path,'Leaving_SB')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'heatmap'))
        
        fig,ax = PlotEventMean(self.tot_reward_signal,event ='Reward Collection',title='Reward collection mean')
        path = os.path.join(output_path,'Reward_Collection')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'Tot mean'))
        
        fig,ax = PlotEventMean(self.tot_leaving_signal,event = 'leaving well',title='leaving well mean')
        path = os.path.join(output_path,'Leaving_well')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'Tot mean'))
        
        fig,ax = PlotEventMean(self.tot_leaving_SB,event = 'leaving SB',title='leaving SB mean')
        path = os.path.join(output_path,'Leaving_SB')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'Tot mean'))
        
        fig,ax = PlotSpeedHistogram(self.tot_velocity['z_score'], self.tot_velocity['speed'])
        path = os.path.join(output_path,'speed')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'Tot speed'))
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax = PlotRadianGraph(self.tot_velocity['z_score'], self.tot_velocity['angle'], ax, label='Original Mean', color='purple')
        #setting up bootstrap analysis
        neo_z_score = []
        for i in range (len(self.tot_velocity['angle'])):
            r = random.randint(0, len(self.tot_velocity['z_score'])-1)
            neo_z_score.append(self.tot_velocity['z_score'].iloc[r])
        ax = PlotRadianGraph(neo_z_score, self.tot_velocity['angle'], ax, label='Bootstrap Mean', color='orange', linestyle='--',title='Total mean of z_score at different head direction')
        ax.legend()
        path = os.path.join(output_path,'angle')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"Tot angle"))
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax = PlotRadianGraph(self.tot_velocity['speed'], self.tot_velocity['angle'], ax, label='Original Mean', color='purple')
        #setting up bootstrap analysis
        neo_z = []
        for i in range (len(self.tot_velocity['angle'])):
            r = random.randint(0, len(self.tot_velocity['speed'])-1)
            neo_z.append(self.tot_velocity['speed'].iloc[r])
        ax = PlotRadianGraph(neo_z, self.tot_velocity['angle'], ax, label='Bootstrap Mean', color='orange', linestyle='--',title='mean of speed at different head direction')
        ax = PlotRadianGraph(self.tot_velocity['z_score'], self.tot_velocity['angle'], ax, label='Original Mean')
        ax.legend()
        path = os.path.join(output_path,'angle')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,f"speed vs angle"))
        
        self.AUCAnalysis()
    
    def AUCAnalysis(self):
        self.AUC_dif_tot = pd.DataFrame()  # Initialize an empty DataFrame
        for day in self.days:
            data = []
            for trail in day.trails:
                if trail.IsImportant:
                    data.extend(trail.AUC_dif_list)  # Collect data from each trail
            # Create a DataFrame for the current day's data and concatenate it
            day_df = pd.DataFrame(data, columns=[f"Day{day.day}"])
            self.AUC_dif_tot = pd.concat([self.AUC_dif_tot, day_df], axis=1)
        fig,ax = PlotDicMean(self.AUC_dif_tot,ylab='AUC_difference',title=mouse_name+' Mean AUC difference')
        path = os.path.join(output_path,'Reward_Collection')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,'AUC_mean'))
        
class group:
    def __init__(self):
        self.mouse = []
        for file in os.listdir(parameter['grandparent_folder']):
            global name
            name = file
            num = re.findall(r'\d+', file)
            if len(num)==0:
                continue
            print('Reading:'+file)
            path = os.path.join(parameter['grandparent_folder'],file)
            global output_path
            output_path  = os.path.join(path,'Neo_output')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.mouse.append(mice(path,file))
            self.mouse[-1].Analysis()
        #Obtain day max for mic
        self.day_max = 9999
        for i in self.mouse:
            day_num = len(i.days)
            if day_num<self.day_max:
                self.day_max = day_num
    
        self.FrequencyPowerAnalysis()
        self.AUCAnalysis()

    def FrequencyPowerAnalysis(self):
        # Collect all data into a list of dictionaries
        data = []
    
        for mouse in self.mouse:
            for day in mouse.days:
                for trail in day.trails:
                    if trail.IsImportant:
                        data.append({'Mouse ID': mouse.mouse_ID, 'Low/High Frequency Ratio': trail.power_ratio})
    
        # Convert the list of dictionaries to a DataFrame
        ratio_df = pd.DataFrame(data)
    
        # Create a boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Mouse ID', y='Low/High Frequency Ratio', data=ratio_df, ax=ax)
    
        # Add a title and labels
        ax.set_title('Low/High Frequency Energy Ratio Boxplot (10Hz cutoff)')
        ax.set_xlabel('Mouse ID')
        ax.set_ylabel('Low/High Frequency Ratio')
        
        # Display the plot
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(os.path.join(parameter['grandparent_folder'], 'Frequency_power_analysis.png'))
    
    def AUCAnalysis(self):
        self.AUC_dif_tot = pd.DataFrame()  # Initialize an empty DataFrame
        for single_mouse in self.mouse:
            self.AUC_dif_tot = pd.concat([self.AUC_dif_tot, single_mouse.AUC_dif_tot], ignore_index=True)
        fig,ax = PlotDicMean(self.AUC_dif_tot,ylab='AUC_difference',title='All mice Mean AUC difference')
        fig.savefig(os.path.join(parameter['grandparent_folder'],'Tot_AUC_mean'))

        
def PlotLinearRegression (x,y,ax):
    x = np.array(x)
    y = np.array(y)
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Generate regression line
    regression_line = slope * x + intercept
    # Plotting the regression line
    ax.plot(x, regression_line, label='Regression line')
    

def PlotEvent (y,ylab = 'z_score',event = 'event', title = 'title'):
    x = np.linspace(-parameter['before_win'], parameter['after_win'], len(y))
    fig, ax = plt.subplots()
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label=event) 
    ax.plot(x, y, label=ylab) 
    ax.set_xlabel('Time')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return fig, ax

def PlotDicMean (data, ylab = 'mean' ,title = 'Mean with Error Bars', confidence=0.95):
    fig, ax = plt.subplots()
    means = {key: np.mean(values) for key, values in data.items()}
    errors = {key: np.std(values, ddof=1) / np.sqrt(len(values)) for key, values in data.items()} 
    # Extract data for plotting
    categories = list(means.keys())
    mean_values = list(means.values())
    error_values = list(errors.values())
    
    ax.errorbar(categories, mean_values, yerr=error_values, fmt='o', capsize=5, linestyle='None', color='black')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    # Adjust the padding between and around subplots for better fit
    plt.tight_layout()
    # Show the plot
    plt.show()
    return fig,ax

def PlotEventMean(df,title = 'mean signal', ylab='z_score',event='event'):
    x = np.linspace(-parameter['before_win'], parameter['after_win'], df.shape[0])
    # Calculate the mean for each index
    means = df.mean(axis=1)
    
    # Calculate the standard error for each index
    standard_errors = df.sem(axis=1)
    
    # Determine the confidence interval
    confidence_level = 0.95
    degrees_freedom = df.shape[1] - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=degrees_freedom)
    
    # Calculate the margin of error
    margin_of_error = t_value * standard_errors
    
    fig, ax = plt.subplots()
    ax.plot(x, means, color='blue', label='Mean Signal')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label=event) 
    ax.fill_between(x, means - margin_of_error, means + margin_of_error, color='blue', alpha=0.2, label='95% CI')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return fig,ax

def PlotEventHeatMap(df,title = 'title'):
    fig, ax = plt.subplots(figsize=(10, 6))
    seaborn.heatmap(df.T,vmax=3,vmin=-3)
    label = np.arange(-parameter['before_win'],parameter['after_win']+1,step=1)
    tick_positions = np.linspace(0, df.shape[0]-1,len(label))
    ax.axvline(x=df.shape[0]/2,color='blue', linestyle='--')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label)
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    return fig, ax

def PlotSpeedHistogram(x, y, title='mean z_score at different speed'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Bins based on y (speed)
    bins = np.linspace(0, min(150, y.max()), int((min(150, y.max()) )/25)+ 1)
    
    # Assign each y value (speed) to a bin
    df['bin'] = pd.cut(df['y'], bins)
    
    # Calculate the mean of x (z_score) for each speed bin
    mean_x_per_bin = df.groupby('bin')['x'].mean()
    
    # Calculate SEM for each bin
    sem_x_per_bin = df.groupby('bin')['x'].apply(sem)  # Standard error of the mean

    # Calculate the confidence intervals for each bin
    confidence_level = 0.95
    degrees_freedom = df.groupby('bin')['x'].count() - 1
    t_value = t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    margin_of_error = t_value * sem_x_per_bin
    
    # Get the bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot the histogram of mean x values for each speed bin
    ax.bar(bin_centers, mean_x_per_bin, width=(bins[1] - bins[0]), color='lightblue', edgecolor='black', alpha=0.7)
    
    # Plot the error bars for the confidence interval
    ax.errorbar(bin_centers, mean_x_per_bin, yerr=margin_of_error, fmt='', color='blue', ecolor='black', capsize=5, label='95% CI')
    # Add labels and title
    ax.set_ylim(mean_x_per_bin.min() - 0.1, mean_x_per_bin.max() + 0.1)
    ax.set_xlabel('Speed (pixel per second)')
    ax.set_ylabel('Mean z_score')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax


def PlotRadianGraph(z, ang, ax=None, label='Mean Signal', color='purple', linestyle='-', alpha=0.2,title='Mean z_score at Different Head Direction'):
    z = np.array(z)
    # Convert angles to radians
    angles_radians = np.deg2rad(ang)
    num_bins = 72
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_indices = np.digitize(angles_radians, bins) - 1

    # Initialize arrays to store mean and CI values
    bin_means = np.zeros(num_bins)
    bin_cis = np.zeros(num_bins)

    # Calculate mean and CI for each bin
    for i in range(num_bins):
        bin_data = z[bin_indices == i]
        if len(bin_data) > 0:
            bin_means[i] = np.mean(bin_data)
            sem_value = sem(bin_data)
            degrees_freedom = len(bin_data) - 1
            t_value = t.ppf(0.975, degrees_freedom)  # 95% CI
            bin_cis[i] = t_value * sem_value
        else:
            bin_means[i] = 0
            bin_cis[i] = 0

    # Repeat the first element to close the circle
    angles_polar = np.linspace(0, 2 * np.pi, num_bins + 1)
    bin_means = np.append(bin_means, bin_means[0])
    bin_cis = np.append(bin_cis, bin_cis[0])

    # Create a new plot or use an existing one
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    else:
        fig = ax.figure

    # Plot the mean signal
    ax.plot(angles_polar, bin_means, color=color, linewidth=2, linestyle=linestyle, label=label)

    # Plot the confidence interval as a shaded area
    ax.fill_between(angles_polar, bin_means - bin_cis, bin_means + bin_cis, color=color, alpha=alpha)

    # Set plot title and legend
    ax.set_title(title)

    return ax
    
def LowPassFilter (x):
    # Sampling frequency (in Hz)
    fs = parameter['atlas_frame_rate'] 
    cutoff = parameter['low_pass_filter_frequency']
    # Normalized cutoff frequency (cutoff frequency / Nyquist frequency)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Order of the filter
    order = 5
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y=signal.filtfilt(b, a, x, axis=0)
    return y
    

a = group()