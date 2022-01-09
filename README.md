# easy_visualization
We provide an easy way for visualizing

You will get a heatmap image when you send it the layer name, the model and the image.

# How
the example code is located in the root, its name is 'Evison_Tutorial.ipynb'

# Online
We provide an example on goolge colab: https://colab.research.google.com/drive/1nNxHOur741WMjBIp1BHrmGi9xHb_CfDB?usp=sharing

# Multi-output
You may specify which output you want to use for classification bythe argument target_output.
```
display = Display(network, visualized_layer, target_output=2, img_size=(224, 224)) # You may use the output of index 3 for classification.
```

# Reference
https://github.com/utkuozbulak/pytorch-cnn-visualizations
# Contact
Jones(Jinhong) Lin: jonneslin@gmail.com / jlin398@wisc.edu
