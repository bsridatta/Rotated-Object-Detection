# Rotated-Object-Detection
Novel ResNet inspired Tiny-FPN network (<2M params) for Rotated Object Detection using 5-parameter Modulated Rotation Loss

### Crux
* **Architecture**: FPN with classification and regression heads ~1.9M parameters  
* **Loss Function**: 5 Parameter Modulated Rotation Loss  
* **Activation**: Mish  
* **Model Summary** - *reports/FPN_torchsummary.txt* (reports/ also contain alterantive summary with named layers in table)
* **Training Script** - *src/train.py* 
* **Final Model Weights** - *src/checkpoints/model_93_ap.pt* 
* **Python Deps. and version** - *requirements.txt*
* **Evaluation** - *src/main.py*


### Method
* The reported results are using a ResNet inspired building block modules and an FPN. 
* Separate classification and regression subnets (single FC) are used. 
* Feature map from the top of the pyramid that has the best semantic representation is used for classification. 
* While the finer feature map at the bottom of the pyramid that has the best global representation is used for regressing the rotated bounding box. Finer details can be found in the code as comments. Code: *src/models/detector_fpn.py* 

* The whole implementation is from scratch, in PyTorch. Only the method for calculating AP from PR curves is borrowed and referenced (*src/metrics.py/compute_ap*). 

### Approach
1. Random data generator that creates images with high noise and rotated objects (shapes) in random scales and orientations. (Private)
2. Compare reusing generated samples for each epoch VS online generating and loading
3. Implement modulated rotated loss and other metrics
4. Experiment with loss functions and activations
5. Tried to replace standard convolutional layers with ORN (Oriented Response Network) that use rotated filters to learn orientation (Could not integrate due to technical challenges)
6. Improve basic model to use different heads for classification and regression
7. Try variations by removing 512-dimensional filters as they take up the most parameters (~1M)
8. Add feature pyramid and experiment with different building blocks and convolutional parameters (kernel size, stride in the first layer plays a big role)
9.  Streamline parameters in the building blocks and the prediction heads to be lower than 2M

* **Please find the rest of the report, with details on experiments and analysis, in** *reports/experiments.pdf* 

### Opportunities to improve
1. Use the rest of the pyramid layers for prediction (take more parameters) and have better logic to get the best detection
2. Integrate ORN layers to FPN
3. Using DenseNets with compact convolution layer configurations
