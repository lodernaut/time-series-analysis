# time-series-analysis
 This tutorial is an introduction to time series forecasting using TensorFlow. It builds several different styles of models, including Convolutional and Recurrent Neural Networks (CNNs and RNNs).

• Single-step forecasting:
    // One feature.
    // All features.
• Multi-step forecasting:
    // Single-shot: Make predictions all at once.
    // Autoregressive: Make one prediction at a time and feed the output back into the model.

• The Meteorological Dataset: (https://www.bgc-jena.mpg.de/wetter/ // https://www.bgc-jena.mpg.de/)

• This dataset contains 14 different features, such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, starting from 2003. For efficiency, we will only use the data collected between 2009 and 2016.

download zip: (https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
