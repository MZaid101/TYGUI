import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import filedialog, Canvas
import tkinter as tk
import cv2


class TrainYolo():
    def __init__(self, root):
        self.root = root
        self.Direc = os.getcwd()
        self.images = []
        self.ClassN = 0
        self.WeightsPath = ""
        self.CFGPath = ""
        self.ObjDataPath = ""
        self.DarknetPath = ""
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.title("TrainYOLO")
        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()-70))
        can1 = Canvas(self.root, width=550, height=215)
        can2 = Canvas(self.root, width=550, height=175)
        can3 = Canvas(self.root, width=950, height=175)
        can4 = Canvas(self.root, width=5, height=100)
        root.bind('Escape',self.ToggleGeom)
        self.btnClr = tk.Button(root, text ='Clear', command = self.Clear, fg="White",  bg="#1285DF", width=0, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnClr.place(x=460, y=20)
        self.txt = tk.Entry(root, width=20 ,fg="Black",bg='#fca72b',font=('times', 10, ' bold '))
        self.txt.place(x=300, y=20)
        self.lblEnter = tk.Label(root, text="Ente Split Percentager: %", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblEnter.place(x=20, y=20)
        self.btnG_TT = tk.Button(root, text ='Generate Train/Test txt', command = self.GenerateTrainTestTxt, fg="White",  bg="#1285DF", width=0, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnG_TT.place(x=20, y=50)
        self.lblClass = tk.Label(root, text="Enter Class Names", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblClass.place(x=20, y=120)
        self.txtClass = tk.Entry(root, width=20 ,fg="Black",bg='#fca72b',font=('times', 10, ' bold '))
        self.txtClass.place(x=300, y=120)
        self.lblClass = tk.Label(root, text="The Format should be: 'class1,class2,class3,..,classN' ", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblClass.place(x=20, y=150)
        self.btnG_CN = tk.Button(root, text ='Generate classes names', command = self.GCN, fg="White",  bg="#1285DF", width=0, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnG_CN.place(x=20, y=180)
        can1.create_rectangle(10, 10, 550, 215, fill="white", outline = 'black')
        can1.place(x=0,y=0)
        
        self.lblCFG = tk.Label(root,text="Configure CFG", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblCFG.place(x=20,y=250)
        self.lblBatch = tk.Label(root,text="Enter batch", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblBatch.place(x=20,y=280)
        self.txtBatch = tk.Entry(root, width=20 ,fg="Black",bg='#fca72b',font=('times', 10, ' bold '))
        self.txtBatch.place(x=300, y=280)
        self.lblSubDiv = tk.Label(root,text="Enter subdivisions", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblSubDiv.place(x=20,y=310)
        self.txtSubDiv = tk.Entry(root, width=20 ,fg="Black",bg='#fca72b',font=('times', 10, ' bold '))
        self.txtSubDiv.place(x=300, y=310)
        self.lblWH = tk.Label(root,text="Enter width/height", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblWH.place(x=20,y=340)
        self.txtWH = tk.Entry(root, width=20 ,fg="Black",bg='#fca72b',font=('times', 10, ' bold '))
        self.txtWH.place(x=300, y=340)
        self.btnCFG = tk.Button(root, text ='Generate obj.cfg', command = self.WriteCFG, fg="White",  bg="#1285DF", width=0, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnCFG.place(x=20, y=370)
        can2.create_rectangle(10, 10, 550, 175, fill="white", outline = 'blue')
        can2.place(x=0,y=230)

        self.lblError = tk.Label(root, text="", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 15, ' bold '), justify=tk.LEFT ) 
        self.lblError.place(x=750, y=60)
        
        
        
        
        self.btnWeights = tk.Button(root, text ='Get Pre Trained Weigths', command = self.GetPreTrainedWeigths, fg="White",  bg="#1285DF", width=20, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnWeights.place(x=10, y=430)
        self.btnDarknet = tk.Button(root, text ='Get Darknet.exe', command = self.GetDarknet, fg="White",  bg="#1285DF", width=20, height=0, activebackground="Red", font=('Helvetica', 10, ' bold '))
        self.btnDarknet.place(x=380, y=430)
        self.btnTrain = tk.Button(root, text ='Start Training', command = self.StartTraining, fg="White",  bg="Red", width=20, height=0, activebackground="#1285DF", font=('Helvetica', 20, ' bold '))
        self.btnTrain.place(x=root.winfo_screenwidth()//2, y=400)
        
        self.lblInfo = tk.Label(root, text="Instructions", width=0, fg="#1245DF",  height=0, activebackground="Red", font=('Helvetica', 15, ' bold '), justify=tk.LEFT ) 
        self.lblInfo.place(x=10, y=530)
        can3.create_rectangle(10, 10, 950,150, fill="white", outline = 'black')
        can3.place(x=0,y=550)
        
        self.lblIns = tk.Label(root, text="Make Sure you Have: \n\t1. CUDA and cuDNN installed. \n\t2. OpenCV built with CUDA and cuDNN. \n\t3. Darknet Configured.", width=0, fg="#DF5421",bg="white",  height=0, font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblIns.place(x=20, y=570)
        self.lblIns = tk.Label(root, text="Check 'TheCodingBug' YouTube Channel to Con-\nfigure OpenCV and Darknet", width=0, fg="#DF5421",bg="white",  height=0, font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblIns.place(x=20, y=640)
        can4.create_rectangle(0, 0, 5,100, fill="black", outline = 'black')
        can4.place(x=350,y=580)
        self.lblIns = tk.Label(root, text="Steps: \n\t1. Generate Train/Test txt. \n\t2. Enter Class Names. \n\t3. Configure CFG whiich should have name: yolov4-custom.cfg. \n\t4. Select Pre-Trained Weights yolov4.conv.137. \n\t5. Select Darknet.exe. \n\t6. Start Training.", width=0, fg="#DF5421",bg="white",  height=0, font=('Helvetica', 10, ' bold '), justify=tk.LEFT ) 
        self.lblIns.place(x=370, y=570)
        
        
    def ToggleGeom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom
    def Clear(self):
        self.txt.delete(0, 'end')
        self.lblError.configure(text='')
        self.txtBatch.delete(0, 'end')
        self.txtSubDiv.delete(0, 'end')
        self.txtWH.delete(0, 'end')
        
    
    def GenerateTrainTestTxt(self):
        if self.txt.get() == '':
            self.lblError.configure(text="Split Percentage is not entered")
            return
        elif self.txt.get() == "100":
            self.lblError.configure(text="Split Percentage cannot be 100%")
            return
        Direc = self.OpenImageDirectory()
        if len(Direc) > 0:
            image_files = []
            os.chdir(os.path.join(Direc))
            for filename in os.listdir(os.getcwd()):
                if filename.endswith((".png", "jpg", "jpeg", ".jfif")):
                    image_files.append(str(os.getcwd())+"\\" + filename)
            os.chdir("..")
            self.Direc = os.getcwd()
            with open("train.txt", "w") as outfile:
                for image in image_files:
                    outfile.write(image)
                    outfile.write("\n")
                outfile.close()
            df=pd.read_csv(str(os.getcwd())+'/train.txt',header=None)
            test_size = float(self.txt.get())/100
            print("Train Data Size: ", round((1-test_size)*100,0), "%")
            print("Test Data Size: ", round((test_size)*100,0), "%")
            data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=test_size, random_state=42)
            data_train=data_train.reset_index()
            data_train=data_train.drop(columns='index')
            with open("train.txt", "w") as outfile:
                for ruta in data_train[0]:
                    outfile.write(ruta)
                    outfile.write("\n")
                outfile.close()
            data_test=data_test.reset_index()
            data_test=data_test.drop(columns='index')
            with open("test.txt", "w") as outfile:
                for ruta in data_test[0]:
                    outfile.write(ruta)
                    outfile.write("\n")
                outfile.close()
            self.txt.delete(0, 'end')
        else:
            self.lblError.configure(text="Error, Directory is Not Selected.")
    
    def GCN(self):
        if self.txtClass.get() == '':
            self.lblError.configure(text="Please Enter Names of Classes")
            return
        else:
            self.lblError.configure(text="")
            text = str(self.txtClass.get())
            count = 0
            classes = text.split(",")
            with open("classes.names", "w") as outFile:
                for line in classes:
                    outFile.write(line)
                    outFile.write("\n")
                    count+=1
            self.ClassN = count
            print("Number of Classes is: ", count)
            print("Classes are: ", classes)
            self.txtClass.delete(0, 'end')
            if not os.path.exists(os.path.join(self.Direc, "backup")):
                os.mkdir(os.path.join(self.Direc, "backup"))
            with open("obj.data", "w") as outFile:
                outFile.write("classes = "+str(count) )
                outFile.write("\n")
                outFile.write("train = "+ os.path.join(self.Direc, "train.txt"))
                outFile.write("\n")
                outFile.write("valid = "+ os.path.join(self.Direc, "test.txt"))
                outFile.write("\n")
                outFile.write("names = "+ os.path.join(self.Direc, "classes.names"))
                outFile.write("\n")
                outFile.write("backup = "+ os.path.join(self.Direc, "backup"))
                outFile.write("\n")
                outFile.close()
            print("obj.data is created.")
            self.ObjDataPath = os.path.abspath("obj.data")
    
    def WriteCFG(self):
        if self.ClassN == 0:
            self.lblError.configure(text="Please Generate Class.names first")
            return
        batchSize = self.txtBatch.get()
        subDiv = self.txtSubDiv.get()
        wh = self.txtWH.get()
        if batchSize == '' or subDiv == '' or wh == '':
            self.lblError.configure(text="Empty Feilds")
            return
        if batchSize != '':
            self.batchSize = int(batchSize)
        if subDiv != '':
            self.subDiv = int(subDiv)
        if wh != '':
            self.wh = int(wh)
        print(self.batchSize, self.subDiv, self.wh)
        self.txtBatch.delete(0, 'end')
        self.txtSubDiv.delete(0, 'end')
        self.txtWH.delete(0, 'end')
        cfgFile = self.OpenFilename()
        cfgF = []
        with open(cfgFile, "r") as cfg:
            for line in cfg:
                cfgF.append(line.rsplit("\n"))
            cfg.close()
        for li in cfgF:
            while '' in li:
                li.remove('')
        mCFG = []
        for line in cfgF:
            if line == ['classes=80']:
                txt = "classes="+str(int(self.ClassN))
                mCFG.append(txt)
            elif line == ['batch=64']:
                txt = "batch="+str(self.batchSize)
                mCFG.append(txt)
            elif line == ['subdivisions=16']:
                txt = "subdivisions="+str(self.subDiv)
                mCFG.append(txt)
            elif line == ['max_batches = 500500']:
                if self.ClassN == 0:
                    pass
                elif self.ClassN <= 2:
                    txt = "max_batches = 6000"
                else:
                    txt = "max_batches ="+str(self.ClassN*2100)
                mCFG.append(txt)
            elif line == ['steps=400000,450000']:
                if self.ClassN == 0:
                    pass
                elif self.ClassN <= 2:
                    txt = "steps = 5800,5900"
                else:
                    txt = "steps="+str(int((self.ClassN*2100)*0.8))+","+str(int((self.ClassN*2100)*0.9))
                mCFG.append(txt)
            elif line == ['width=608']:
                txt = "width="+str(self.wh)
                mCFG.append(txt)
            elif line == ['height=608']:
                txt = "height="+str(wh)
                mCFG.append(txt)
            elif line == ['filters=255']:
                if self.ClassN == 0:
                    pass
                else:
                    txt = "filters="+str((5+self.ClassN)*3)
                mCFG.append(txt)
            else:
                mCFG.append(line)
        with open("obj.cfg","w") as cfgFile:
            for line in mCFG:
                cfgFile.write("".join(line))
                cfgFile.write("\n")
        self.CFGPath = os.path.abspath("obj.cfg")
    
    def GetPreTrainedWeigths(self):
        weights = self.OpenFilename()
        if weights.endswith(".conv.137"):
            self.WeightsPath = os.path.abspath(weights)
            self.lblError.configure(text="Weights File Loaded")
        else:
            self.lblError.configure(text="Invalid Weights File")
    
    def GetDarknet(self):
        darknet = self.OpenFilename()   
        if darknet.endswith("darknet.exe"):
            self.DarknetPath = os.path.abspath(darknet)        
            self.lblError.configure(text="Darknet Loaded")
            print(self.DarknetPath)
        else:
            self.lblError.configure(text="Invalid File")
         
    def StartTraining(self):
        if cv2.cuda.getCudaEnabledDeviceCount() == 0 :
            self.lblError.configure(text="GPU is not Found/Enabled\nPlease Use Google Colab Instead")
            return
        else:
            if self.CFGPath == "" or self.DarknetPath == "" or self.ObjDataPath == "" or self.WeightsPath == "":
                self.lblError.configure(text="Error, Please Do Initial Steps First")
            else:
                self.lblError.configure(text="Starting Training...")
                data = self.ObjDataPath
                cfg = self.CFGPath
                weights = self.WeightsPath
                darknet = self.DarknetPath.replace("\\darknet.exe","")
                print(darknet)
                print(cfg)
                print(weights)
                print(data)
                with open("Darknet.bat",'w') as darknetFile:
                    darknetFile.write("C:\n")
                    darknetFile.write("cd C:/Users/Dell\n")
                    darknetFile.write("cd {0}\n".format(darknet))
                    darknetFile.write("darknet.exe detector train {0} {1} {2} -map".format(data,cfg,weights))
                    darknetFile.close()
    
                os.system(r"Darknet.bat")

    
    
    def OpenImageDirectory(self):
        Direc = filedialog.askdirectory(title= 'Select Directory')
        return Direc
    
    def OpenFilename(self):
    	self.filename = filedialog.askopenfilename(title ='Select CFG File')
    	return self.filename
    


