# Acquisition Program
This acquisition program is for "Finger Movement Recognition via High-Density Electromyography of Intrinsic and Extrinsic Hand Muscles"
 * The acquisition program is written by Python 3.8 and compatible with Python 3.6
 * UI is designed by PyQt5, 
### Instruction
The collected gestures are as follows:

| 1 finger | 2 and 3 fingers      | 4 fingers and others   |  
|  ----    | ----                 |       ----             |  
|01 index  |05 index+middle       |  10 fingers flex       |                    
|02 middle |06 middle+ring        |  11 fingers abduction  |               
|03 ring   |07 ring+little        |   12 fingers adduction |                            
|04 little |08 index+middle+ring  |                        |
|          |09 middle+ring+little |                        |

#### To run the program, you should first install some necessary packages.

1. Run "pip install -r requirements.txt" in your terminal.
2. Open Myo Connect.exe before running the acquisition program.<sup>1</sup>
3. Run "python Program.cpython-36.pyc" to open the acquisition program.

#### To start the program, the steps are as follows:
1. Tap the "Add User" button to input the subject information.
2. Select the right subject in the below Combo Box, then tap "Load" button to load the data.<sup>2</sup>
3. Select the right COM Port to connect to the Data Glove.
4. Set the "Myo Connect.exe" to connect the MYO armband.
5. Connect the OT device with your PC.
5. Tap "Start" button to start the acquisition.<sup>3</sup>

#### To collect the data of the specific gesture, you can:
1. Press "up" and "down" keys to choose the gesture you want to make.
2. Press "Space" key to start the collecting, then the Target Trajectory Guidance Experiment will start. <sup>4</sup>
3. You can press the "Space" again to restart the collection.
4. After ten seconds, the collection is stop, you can choose "ok" or "cancel" to accept or reject this trial.

Finally, at the end of the experiment, tap "Stop" to end the communication, then close the program at upper right icon

Note:
1. If you skip or forget the step.2, it will pop up an error , and also reminds you to open official "MYO Connect.exe"
2. Don't worry if you find the program is not responding after tapping the "Loading" button. 
   It is because loading the data needs time, especially when you change to another subject who have just completed a session of data acquisition. 
   More data were collected, it will need more time to load them, but not too long.
3. you can choose to watch the specific sensor data by selecting the corresponding box in the "Debug" zone, 
   so you don't have to open all the devices.
4. You'd better not do Step. 1 now, otherwise the program may go wrong
