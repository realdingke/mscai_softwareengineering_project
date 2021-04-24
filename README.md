# MscAI Software Engineering Group Project (Group 6)
This is the Imperial College Msc AI group project on multi-object tracking, originating from a proposal made by Imperial's industrial partner Cord AI which also grants access to the object tracking datasets stored on its own Cord platform. 

The project essentially rests upon ifzhang's fundamental approach of FairMOT (Github homepage is [here](https://github.com/ifzhang/FairMOT); paper: [FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/pdf/2004.01888v5.pdf)), linking up with the Cord's database and constructing an highly automated pipeline that can enable user to either train new models from scratch or track with existing models (from after the training scheme or pre-trained model downloadable from public URL). It is also worth noting the training-tracking pipeline possesses full multi-class capabilities (multi-class training & evaluation), building onto the powerful MCMOT modifications by CaptainEven whose homepage is linked here: [FairMOTVehicle](https://github.com/CaptainEven/FairMOTVehicle).

### Some demos/highlights
Highlights on traffic datasets are below:
![Highway traffic](demos/Highway_traffic.gif)

Highlights on visdrone datasets are below:
![Cattle](demos/cattle.gif)

### Overall structure of the pipeline
![pipeline structure overview](demos/pipeline_structure_overview.jpg)


### Means of running the pipeline: CLI or API based 
The pipeline is designed to be run via two approaches: traditional CLI approach or the more user-friendly API.

### - [x] CLI approach:
The below are instructions for running the pipeline via a CLI.

### Entry Point
The entry point is located inside the `/src` folder which also corresponds to the default root directory of the program.

The first step always is to run the entry point file with `--gen_info` flag to see the important dataset information, facilitating user's decision to run desired pipeline branch with appropriate data and model:

    python3 entry_point.py --gen_info


### Training
Under `/src` (root dir) run:

    python3 entry_point.py --train_track

Using the `--train_track` flag will train a new model from sratch with architecture, learning rate, epoch number, batch size etc. of user's choosing.


### Tracking
Under `/src` (root dir) first run:

    python3 entry_point.py --track

specifying the `--track` flag, then run:

    python3 mctrack.py

This will run multi-class tracking with a desiginated model on assigned data/video. Note multi-class evaluation is embedded inside the tracking scheme thus will also be invoked by the tracking method, producing a spreadsheet file containing comprehensive object tracking results.

