
- `Segmentation_YOLO/`
    - `data_pro_YOLO_Single_inside.py`
    - `data_pro_YOLO_Single_outside.py`
    - `evaluation_yolo.py`

--- 
--- 

### **Segmentation-YOLO**

- ###### **Data Process**

 ```
  python data_pro_YOLO_Single_inside.py
  python data_pro_YOLO_Single_outside.py
  ```

  Entering the commands mentioned above in the terminal will create folders similar to 'BRICK_data_YOLO_Single', which can be used as training data.

- ###### **Training Models**

  In the folder BRICK_data_YOLO_Single, there is a data.yaml file contained a command 

  ```
  yolo segment train data=/scratch/tz2518/Segmentation_YOLO/BRICK_data_YOLO_Single/data.yaml model=yolov8x-seg.yaml pretrained=/scratch/tz2518/ultralytics/yolov8x-seg.pt epochs=1000 imgsz=1024 cache=True name=BRICK
  ```

  copy this line and paste in the terminal to run the training of BRICK. A similarly prepared command can be used in the data for other features.yaml files.

- ###### **Get the Output Folder**

  After returning the command at the last step, folders named by feature names are created in the folder-runs/segment. It may take a few hours to train the models. When it is complete, the folder like-runs/segment/BRICK will contain a few images as the output and predictions. There will also be a folder -runs/segment/BRICK/weights containing the model's weight.

- ###### **Evaluate the model**

  Run the evaluation code using this command

  ```
  python evaluation_yolo.py
  ```

  An Excel will be got in the same folder of evaluation_yolo.py. It calculates each feature's TP,  TN,  FP, FN, Accuracy, Precision, and Recall.
