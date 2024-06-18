# Four-dimensional label-free live cell image segmentation

This is the code for [Four-dimensional label-free live cell image segmentation for predicting live birth potential of embryos]().
This project is carried out in cooperation with [Funahashi Lab. at Keio University](https://fun.bio.keio.ac.jp/) and Yamagata Lab. at Kindai University.

<img src="images/segmentation_result.gif" alt="result" width="600"/>

## Overview

Our method performs instance segmentation of 3D bright-field microscopy images.
Our model performs instance segmentation of the time-series 3D bright-field microscopy images at each time point, and the quantitative criteria for mouse development are extracted from the acquired time-series segmentation image.

<img src="images/model.png" alt="model" width="500"/>


## QuickStart

1. Download this repository by `git clone`.

   ```sh
   % git clone https://github.com/funalab/BFsegmentation.git
   ```

2. Install requirements.

   ```sh
   % cd BFsegmentation/
   % python -m venv venv
   % source ./venv/bin/activate
   % pip install -r requirements.txt
   ```

3. Inference on example test dataset.
  
   Currently we provide some pretrained models for 3d and 3d+t bright-feild microscopy image.
   * `models/best_model.pth` : model trained using `confs/model/base.yaml`
   * `models/best_model_gru3.path` : model with GRU(Gated recurrent unit) trined using `confs/model/gru3.yaml`

   ```sh
   % CUDA_VISIBLE_DEVICES=1 python src/test.py \
      --test_conf confs/test/test.yaml \
      --model_conf confs/model/base.yaml \
      -o results/test \
      -m models/best_model.path \
      --save_img
   ```

## Training and Inference on your data
   This system uses [AttrDict](https://github.com/bcj/AttrDict) to access config values as attribute.
   For training and inference on your data, set config with reference to `confs/runtime/base.yaml` and `confs/test/test.yaml`.
   
   * runtime/inference config (e.g. `confs/runtime/base.yaml` / `confs/test/test.yaml`)
     * `cfg.DATASETS.DIR_NAME.RAW` : Specify directory path of raw images.
     * `cfg.DATASETS.DIR_NAME.INSTANCE` : Specify directory path of ground truth images for instance segmentation.
     * `cfg.DATASETS.SPLIT_LIST` : Specify the path of the file in which the input file name used for training/validation/test is enumerated.
     * `cfg.DATASETS.RESOLUTION` : Specify microscopy resolution of x-, y-, and z-axis. (defalt=1.0:1.0:2.18)

   * model config  (e.g. `confs/model/base.yaml` / `confs/model/gru3.yaml`)  
  
   **NOTE**: The pair of image and ground truth must be the same name `T.tif` (T: zero-based temporal index composed of three digits).  
   **NOTE**: For time-series images, `confs/model/gru3.yaml` is recommended for higher performance. If you want to perform time-independent segmentation, `confs/model/base.yaml`, which offers a lighter model, is recommended.

1. **Training**:  
   Run the following command to train segmentation model on the datasets/input_example dataset.
   The training results will be generated in the `results/train_[time_stamp]` directory, and
   trained models will be stored sequentially in the `results/train_[time_stamp]/trained_models`.

   ```sh
   % CUDA_VISIBLE_DEVICES=[GPU ID] python src/train.py \
      --runtime_conf confs/runtime/base.yaml \
      --model_conf confs/model/base.yaml \
      -o results/train
   ```

2. **Validation**:  
   Pass the directory path that stores the models genereted by the training process (`results/train_[time_stamp]/trained_models`) to the argument `-m`.
   Run the following command to validate generated models and get best model.
   ```sh
   % CUDA_VISIBLE_DEVICES=[GPU ID] python src/validation.py \
      --test_conf confs/test/validation.yaml \
      --model_conf confs/model/base.yaml \
      -m [path/to/models/directory] \
      -o results/val
   ```
   
3. **Test**:  
   Pass the best-model path selected by validation process to the argument `-m`.
   The segmentation images will be generated in the `results/test_[time_stamp]/Predictions`.
   If you want to evaluate segmentation accuracy, use the argument `--evaluation`.
   ```sh
   % CUDA_VISIBLE_DEVICES=[GPU ID] python src/test.py \
      --test_conf confs/test/test.yaml \
      --model_conf confs/model/base.yaml \
      -m [path/to/trained/model] \
      -o results/test \
      --save_img
   ```

4. **Extraction of quantitative criteria of time-series data**:  
   Pass the directory path that stores the segmentation images outputted by process 3 (`results/test_[time_stamp]/Predictions`) to the argument `-i`.
   Extracted quantitative criteria will be exported to `criteria.csv`.
   ```sh
   % python -i [path/to/segmentation/images] -o results/feat
   ```
 
## Acknowledgement

The microscopy images included in this repository is provided by Yamagata Lab., Kindai University.
The development of this algorithm was funded by JSPS KAKENHI Grant Numbers 16H04731 and 20H03244 to [Akira Funahashi](https://github.com/funasoul).

## References

<a name="ref1"></a> [[1] Schmid, Benjamin, et al. "A high-level 3D visualization API for Java and ImageJ." BMC bioinformatics 11.1 274 (2010).](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-274)