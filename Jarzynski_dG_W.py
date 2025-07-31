#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:16:42 2025

@author: w2040021
"""

import numpy as np
import pandas as pd

import sys
import json
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
import matplotlib.pyplot as plt
import csv
plt.style.use('dark_background')
from IPython.display import display,HTML
display(HTML("<style>.container{width:95% !important;}</style>"))
import scipy


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QSlider, QPushButton, QLCDNumber
from PyQt5.QtCore import QSize
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit





qtCreatorFile = "UI_Jarzynski_dG_W.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = loadUiType(qtCreatorFile)        

Total_Work = []
Work_polymer = []

global file_path
file_path = ""

def WLC(x, Lc, lp, L0):
    return (4.11 / lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    


    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        super().__init__()
        
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')
        
        # self.Pulse_slider.setValue(700)
        # self.LCD1.display(700)
        # self.Pulse_slider.valueChanged.connect(self.LCD1.display)
        # self.Pulse_num.setText('0') #MOhm 
        # self.Cm.setText('10') #pFarads
        # self.gK.setText('8e-7') #S
        # self.gNa.setText('1e-6') #S
        # self.Vrest.setText('-77') #mV
        self.Open_JSON.clicked.connect(self.File_open)
            

        ## SELF BUTTONS 
        self.Plot_button.clicked.connect(self.Plotter)
        self.Plot_avg_traj.clicked.connect(self.Integrate_curve)
        self.Step.clicked.connect(self.WLC_work)
        #self.Histo.clicked.connect(self.Do_histogram)
        self.Save_file.clicked.connect(self.Save_to_file)
        self.Save_work.clicked.connect(self.Save_works)
   
        grafico=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(grafico)
        self.grafico=grafico.figure.subplots()
        self.grafico.set_xlabel('Extension (nm)', size=8)
        self.grafico.set_ylabel('Force (pN)', size=8)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)

        # initial guess for gui 
        self.Fmin_2.setValue(4.5)
        self.Fmin.setValue(10)
        self.Fmax_2.setValue(7.8)
        self.Fmax.setValue(24)
        self.Guess_4.setValue(50.0) 
        self.Guess_5.setValue(0.5)  
        self.Guess_6.setValue(0.0) 
        self.Guess_1.setValue(110.0)  
        self.Guess_2.setValue(0.5)
        self.Guess_3.setValue(10.0) 

        # collego save_df a Wclean
        self.save_df.clicked.connect(self.Wclean)
        # bottone per W tot + hist W tot
        self.compute_sgW.clicked.connect(self.sg_Wtot)

        grafico_hist = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True))
        self.W_hist.layout().addWidget(grafico_hist)  
        self.grafico_hist = grafico_hist.figure.subplots()

        #grafico3=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        #self.Plotting_box_3.addWidget(grafico3)
        #self.addToolBar(NavigationToolbar(grafico, self))
        #self.grafico3=grafico3.figure.subplots()
        #self.grafico3.set_xlabel('Unfolding Force (pN)', size=8)
        #self.grafico3.set_ylabel('Counts', size=8)
        
  
    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
        
        return file_path
      
   
      
    def Load_JSON(self):        
        with open(file_path , 'r') as f:
           d=json.load(f) 
         
      #  d = self.File_open()  
        Pulse_num=self.Pulse_num.value()
        xs, forces, times, current = [], [], [], []
        for i in range (len(d)):
            xs.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["z"]))
            forces.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["force"]))
            times.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["time"])-d["Pulse_Number_"+str(Pulse_num)]["time"][0])    
            current.append(np.array(d["Pulse_Number_"+str(Pulse_num)]["current"]))
            
            
        return xs, forces, times, current
        
        
    def Plotter(self, plot_avg=False):
        
        # xs, forces, times, current=self.Load_JSON()
        # Pulse_num=self.Pulse_num.value()
        
        # xs_smth = []
        
        # xs_smth= savgol_filter(xs, 51, 4)
        
        # i=Pulse_num
        

        # forces[i] = (3.48e-5)*current[i]**2-0.0029*current[i]           # costanti per sisteare il valore della F (non era giusot dai dati)
        
        # self.grafico.clear()
        # self.grafico.set_xlabel('Extension (nm)', size=6)
        # self.grafico.set_ylabel('Force (pN)', size=6)
        # self.grafico.plot(xs[i], forces[i], alpha=0.5)
        # self.grafico.plot(xs_smth[i], forces[i], alpha=0.5)
        # self.grafico.figure.canvas.draw()
        
        # return Pulse_num, xs_smth, forces, times

        xs, forces, times, current = self.Load_JSON()
        Pulse_num = self.Pulse_num.value()

        xs_smth = savgol_filter(xs, 51, 4)

        for i in range(len(forces)):
            forces[i] = np.array((3.48e-5) * current[i]**2 - 0.0029 * current[i])

        self.grafico.clear()
        self.grafico.set_xlabel('Extension (nm)', size=6)
        self.grafico.set_ylabel('Force (pN)', size=6)

        if plot_avg:
            x_avg, f_avg = self.avg_traj(xs_smth, forces, n=5)
            self.grafico.plot(x_avg, f_avg, '-', lw=1, alpha=0.5,  color='orange')
        else:
            self.grafico.plot(xs[Pulse_num], forces[Pulse_num], alpha=0.5)
            self.grafico.plot(xs_smth[Pulse_num], forces[Pulse_num], alpha=0.5)
        self.grafico.figure.canvas.draw()

        return Pulse_num, xs_smth, forces, times

    
    
    def avg_traj(self, xs_smth, forces, n=5):
            min_len = min(len(xs_smth[i]) for i in range(n))
            xs_array = np.array([xs_smth[i][:min_len] for i in range(n)])
            fs_array = np.array([forces[i][:min_len] for i in range(n)])
            x_avg = np.mean(xs_array, axis=0)
            f_avg = np.mean(fs_array, axis=0)

            return x_avg, f_avg
    
    # load for each sg work tot
    def Load(self):
        with open(file_path, 'r') as f:
            d = json.load(f)

        xs, forces_calc, times, current = [], [], [], []

        for i in range(len(d)):
            key = f"Pulse_Number_{i}"

            # estrazione base
            x = np.array(d[key]["z"])
            t = np.array(d[key]["time"]) - d[key]["time"][0]
            I = np.array(d[key]["current"])

            # calcolo forza teorica per ogni pulse
            F = (3.48e-5) * I**2 - 0.0029 * I

            # salvataggio
            xs.append(x)
            times.append(t)
            current.append(I)
            forces_calc.append(F)

        return xs, forces_calc, times, current


    def sg_Wtot(self):
        xs, forces, times, current = self.Load()
        work_list = []

        xs_smth = [savgol_filter(x, 51, 4) for x in xs]

        for idx in range(min(5, len(forces))):
            xmin = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > 6 and forces[idx][i - 1] < 6)
            xmax = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > 20 and forces[idx][i - 1] < 20)

            integrate_force = forces[idx][xmin:xmax]
            integrate_x = xs_smth[idx][xmin:xmax]
            W = scipy.integrate.simpson(integrate_force, integrate_x)
            work_list.append(W)

        # self.Label_TotalWork.setText(",".join(f"{w:.2f}" for w in work_list))
        self.grafico_hist.set_xlabel('Total Work  [pNnm]', size=4)
        self.grafico_hist.hist(work_list, bins=20, edgecolor='black', alpha=0.6, color='orange')
        self.grafico_hist.figure.canvas.draw()

        return work_list



    def Integrate_curve(self):
        Pulse_num, xs_smth, forces, times = self.Plotter(plot_avg=True)  # togli contenuto quando vuoi plot sg 
        x_avg, f_avg = self.avg_traj(xs_smth, forces, n=5)
        self.grafico.plot(x_avg, f_avg, '-', lw=2, color='orange')

        # == W on selected pulse==
        # Work = 0
        # for i in range(len(forces[Pulse_num]) - 1):
        #     if forces[Pulse_num][i] > 6 and forces[Pulse_num][i - 1] < 6:
        #         xmin = i
        #     if forces[Pulse_num][i] > 20 and forces[Pulse_num][i - 1] < 20:
        #         xmax = i

        # integrate_force = forces[Pulse_num][xmin:xmax]
        # integrate_x = xs_smth[Pulse_num][xmin:xmax]
        # Work = scipy.integrate.simpson(integrate_force, integrate_x)
        # xmin_reale = xs_smth[Pulse_num][xmin]
        # xmax_reale = xs_smth[Pulse_num][xmax]
        # print ("xmin real:", xmin_reale, "xmax_reale:", xmax_reale)

        # == calcolo limits si traj media == 
        for i in range(1, len(f_avg)):
            if f_avg[i] > 6 and f_avg[i - 1] < 6:
                xmin = i
            if f_avg[i] > 20 and f_avg[i - 1] < 20:
                xmax = i

        xmin_reale = x_avg[xmin]
        xmax_reale = x_avg[xmax]
        print("xmin reale:", xmin_reale, "xmax reale:", xmax_reale)

        return xmin_reale, xmax_reale, x_avg, f_avg

    def WLC_work(self):
        def compute_WLC_dx(self, xmin_reale, xmax_reale, x_avg, f_avg):

            # Step 1: Limiti per il fitting
            limit_min = self.Fmin.value()  # es. 12 pN
            limit_max = self.Fmax.value()  # es. 20 pN

            ## == versione SINGLE PULSE==
            # for i in range(1, len(xs_smth[Pulse_num])):
            #     if forces[Pulse_num][i] > limit_min and forces[Pulse_num][i - 1] < limit_min:
            #         i_min = i
            #     if forces[Pulse_num][i] > limit_max and forces[Pulse_num][i - 1] < limit_max:
            #         i_max = i

            # xfit = xs_smth[Pulse_num][i_min:i_max]
            # yfit = forces[Pulse_num][i_min:i_max]

                ## == versione MULTI PULSE == 
            for i in range(1, len(f_avg)):
                if f_avg[i] > limit_min and f_avg[i - 1] < limit_min:
                    i_min = i
                if f_avg[i] > limit_max and f_avg[i - 1] < limit_max:
                    i_max = i
            xfit = x_avg[i_min:i_max]
            yfit = f_avg[i_min:i_max]

            guess = [self.Guess_1.value(), self.Guess_2.value(), self.Guess_3.value()]
            popt, _ = curve_fit(WLC, xfit, yfit, p0=guess, bounds=([50, 0.15, 0], [120, 0.8, 60]))
            Lc, Lp, L0 = popt

            x_interp = np.linspace(xmin_reale, xmax_reale, 1000)  
            x_vals = []
            y_vals = []

            for x in x_interp:
                try:
                    F = WLC(x, Lc, Lp, L0)
                    x_vals.append(x)
                    y_vals.append(F)
                except:
                    continue

            area_wlc = abs(scipy.integrate.simpson(y_vals, x_vals))
            # fig, ax = plt.subplots()
            # ax.plot(x_vals, y_vals, '--r', label="DEBUG esteso")
            # ax.legend()
            # plt.show()
            self.grafico.plot(x_vals, y_vals, '--', lw=0.8, color='red', label='WLC extrapolation')
            self.grafico.relim()
            self.grafico.autoscale_view()
            self.grafico.figure.canvas.draw()

            return area_wlc, popt

    
        def compute_WLC_sx(self, xmin_reale, x_avg, f_avg):
            f_min = self.Fmin_2.value()  # 5 pN
            f_max = self.Fmax_2.value()  #  9 pN

            # FIT RANGE

            ## == verisone SINGLE PULSE ==
            # for i in range(1, len(xs_smth[Pulse_num])):
            #     if forces[Pulse_num][i] > f_min and forces[Pulse_num][i - 1] < f_min:
            #         idx_min = i
            #     if forces[Pulse_num][i] > f_max and forces[Pulse_num][i - 1] < f_max:
            #         idx_max = i

            # xfit_sx = xs_smth[Pulse_num][idx_min:idx_max]
            # yfit_sx = forces[Pulse_num][idx_min:idx_max]

            ## == versione MULTI PULSE == 
            for i in range(1, len(f_avg)):
                if f_avg[i] > f_min and f_avg[i - 1] < f_min:
                    idx_min = i
                if f_avg[i] > f_max and f_avg[i - 1] < f_max:
                    idx_max = i
            xfit_sx = x_avg[idx_min:idx_max]
            yfit_sx = f_avg[idx_min:idx_max]

            # Elimino valori vicini a Lc per evitare divergenze
            guess_sx = [self.Guess_4.value(), self.Guess_5.value(), self.Guess_6.value()]
            mask = xfit_sx < 0.95 * guess_sx[0]
            xfit_sx = xfit_sx[mask]
            yfit_sx = yfit_sx[mask]

            popt_sx, _ = curve_fit(WLC, xfit_sx, yfit_sx, p0=guess_sx, bounds=([20, 0.15, 0], [80, 0.8, 60]))
            Lc, Lp, L0 = popt_sx
            yfit_curve = WLC(xfit_sx, Lc, Lp, L0)

            x  =xmin_reale
            x_vals_sx = []
            y_vals_sx = []
            while x > -50:  # estendiamo anche in negativo
                try:
                    f = WLC(x, Lc, Lp, L0)
                except:
                    break
                if f < 0.1:
                    x_zero = x
                    break
                x_vals_sx.append(x)
                y_vals_sx.append(f)
                x -= 0.05

            x_vals_sx = x_vals_sx[::-1]
            y_vals_sx = y_vals_sx[::-1]

            area_wlc = scipy.integrate.simpson(y_vals_sx, x_vals_sx)
            self.grafico.plot(x_vals_sx, y_vals_sx, '--', lw=0.9, color='blue', label='WLC sx extrapolated')
            self.grafico.plot(xfit_sx, yfit_curve, '--', lw=0.9, color='lightblue', label='WLC fit sx')
            self.grafico.relim()
            self.grafico.autoscale_view()
            self.grafico.figure.canvas.draw()

            return area_wlc, popt_sx

    
        # work_dx, params_dx = compute_WLC_dx( xmin_reale)
        # work_sx, params_sx = compute_WLC_sx( xmin_reale)
        xmin_reale, xmax_reale, x_avg, f_avg = self.Integrate_curve()
        work_dx, params_dx = compute_WLC_dx(self, xmin_reale, xmax_reale, x_avg, f_avg)
        work_sx, params_sx = compute_WLC_sx(self, xmin_reale, x_avg, f_avg)


        self.Label_WLCfit_dx.setText(
        f"Lc_dx={params_dx[0]:.2f}  Lp_dx={params_dx[1]:.2f}  L0_dx={params_dx[2]:.2f}"
    )
        self.Label_WLCfit_sx.setText(
        f"Lc_sx={params_sx[0]:.2f}  Lp_sx={params_sx[1]:.2f}  L0_sx={params_sx[2]:.2f}"
    )

        # self.Label_TotalWork.setText(f"Total Work={self.work_list:.2f} pNnm")
        self.Label_WorkWLC_dx.setText(f"Work WLC dx={work_dx:.2f}")
        self.Label_WorkWLC_sx.setText(f"Work WLC sx={work_sx:.2f}")

        # df = pd.DataFrame({
        #     "Region": ["Numerical Total", "WLC DX ", "WLC SX "],
        #     "Work (pNnm)": [Work, work_dx, work_sx]
        # })
        # print(df)

        return work_dx, work_sx
    

    # salvo tutti i valori in df 
    def Wclean(self):
        dfs = []
        for pulse in range(5):
            self.Pulse_num.setValue(pulse)
            QtWidgets.QApplication.processEvents()  

            work, work_dx, work_sx = self.Integrate_curve()
            wclean = work - work_dx + work_sx

        # work, work_dx, work_sx = self.Integrate_curve()
        # wclean = work - work_dx + work_sx 
            pulse_num = self.Pulse_num.value()

            df = pd.DataFrame([{
            "Pulse": pulse_num,
            "W_tot": work,
            "W_wlc_dx": work_dx,
            "W_wlc_sx": work_sx,
            "W_clean": wclean
            }])
            dfs.append(df)
        df_tot = pd.concat(dfs, ignore_index=True)
        df_tot.to_pickle("wclean_results.pkl")
        print(df_tot) 
    
    # store output di ogni pulse
    def Save_works(self):
        Work, Work_WLC=self.Integrate_curve()
        
        Total_Work.append(Work)
        Work_polymer.append(Work_WLC)
        print (Total_Work)
    
        #def Do_histogram(self):
        
        #   F,A= self.Integrate_curve()

        #   hist, bins = np.histogram(Unfolding_forces, range = np.min(Unfolding_forces))

            #self.grafico3.clear()
            #self.grafico2.set_xlabel('Time (s)', size=8)
            #self.grafico2.set_ylabel('Force (pN)', size=8)
            #self.grafico3.hist(F)
            #self.grafico3.figure.canvas.draw()

    
        # salva tutti i values stored con save works in un file csv 
    def Save_to_file(self):
        #F= self.Step_detect()

        np.savetxt("Total_work.csv", Total_Work,  delimiter=",")
        np.savetxt("Work_polymer.csv", Work_polymer,  delimiter=",")
    
        
    def remove_value(self):
        del Total_Work[-1]
        del Work_polymer[-1]


if __name__ == "__main__":
     app = QtWidgets.QApplication(sys.argv)
     window = MyApp()
     window.show()
     sys.exit(app.exec_())   