# Object detection trainer

### finetune object detection models on your own data

More models and dataset formats will be added in the future but for now these are available

#### Supported models:

- fasterrcnn_resnet50_fpn

#### Supported dataset formats :

- MS COCO JSON

### FIRST AND FOREMOST ANNOTATE A DATASET OR GET A DATASET TO WORK WITH AND PLACE IT IN THE 'data' DIRECTORY THEN IT SHOULD LOOK LIKE THIS

```
data/
├── train/
├── test/
└── valid/
```

### installation

This was made with python 3.11 if you're on another version and it doesn't work consider using python 3.11

- create a python virtual env and activate it

```
python -m venv .venv
cd .venv
cd Scripts
activate
cd ..
cd ..
```

- install pytorch with cuda 12.4

```
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- install other requirements.txt

```
pip install -r requirements.txt
```

### Running the scripts

- you can use the default hyper parameters or open config.yaml and change them

- To train, log metrics, and test a model

```
python main.py
```

- To only test a trained model

```
cd src
python tester.py --model-name=your_model_name_here
# model names are timestamps which will be a number that looks like 1742073877
```

### Now check the tests directory and you'll find it filled with comparisons between the model's prediction and the ground truth 

- view logs in tensorboard

```
tensorboard --logdir=logs/
```
