# Applied Machine Learning Overview
Applied machine learning using ONNX Runtime enables the creation of machine learning models specifically for web applications. This integration enhances user experience by combining software engineering with data science, allowing for smarter and more responsive products.

## Challenges for Indie Developers
The modern app market has become increasingly competitive, particularly for indie developers. With dominant platforms like TikTok and Instagram leading through data-driven features, leveraging machine learning has become essential. This shift has made it harder for independent apps to thrive, pushing many developers toward corporate roles.

## Web vs. Mobile App Development
Developing smart applications involves selecting suitable technologies, whether cross-platform frameworks like Flutter and React Native or native tools like Xcode and Java. The web has emerged as a powerful option, offering native-like performance, seamless responsiveness, and easier cross-device deployment, making it ideal for integrating machine learning capabilities.

## Convergence of Skills
Web developers and machine learning engineers often come from distinct educational and technical backgrounds. While web developers focus on usability and interactivity, machine learning engineers excel at data handling and model building. Bridging these disciplines is critical for building ML-powered web applications.

## Machine Learning Tools
Popular machine learning tools include TensorFlow, known for its robust open-source ecosystem. TensorFlow.js brings this power to JavaScript environments, while Brain.js offers a lightweight alternative for simpler ML tasks. ONNX, backed by Microsoft, benefits from cross-industry collaboration and is well-suited for deploying models across platforms.

## TensorFlow.js and Demos
Using the ONNX Runtime via npm packages allows integration of pre-trained models into web environments. TensorFlow.js supports in-browser training and transfer learning. Demos such as the Emoji Scavenger Hunt showcase image recognition tasks and make machine learning interactive and approachable.

## ML5.js and Magenta.js
ML5.js, built on TensorFlow.js, enables browser-based machine learning tasks like image and sound classification. Magenta.js focuses on creative applications such as generating music and art. Both libraries offer accessible interfaces for experimentation and artistic expression through AI.

## Brain.js Overview
Brain.js enables neural networks to run in both browsers and Node.js. It supports tasks like text generation and basic image processing. While it has a simpler API and fewer features compared to TensorFlow.js, it is well-suited for beginners exploring machine learning in JavaScript environments.

Training Models in Python
Python remains the primary environment for training machine learning models, with tools like Jupyter Notebooks and Python scripts offering flexibility and control. Platforms such as Loeb.ai simplify training processes, especially for image recognition. Not all tasks require deep learning; traditional machine learning algorithms often suffice.

Using Scikit-Learn
Scikit-learn provides powerful classic machine learning algorithms with excellent documentation and ease of use. It supports a wide range of models and is ideal for developers building intelligent systems without needing deep learning frameworks.

## Building a Cuisine Recommender
A cuisine recommendation system can be built using scikit-learn, where users input available ingredients and receive cuisine suggestions. The system is trained on a dataset of recipes, undergoing steps like data cleaning and class balancing, and is later deployed in a web app using ONNX Runtime.

## Model Evaluation and Accuracy
Different classifiers were tested to assess model performance. Logistic regression achieved 71% accuracy for predicting Indian recipes, while Support Vector Classifier (SVC) improved accuracy to 83%. The final model was converted using skl2onnx and visualized with Netron for deployment readiness.

## Integrating Model into Web App
A web application was developed using the ONNX Runtime CDN. Users select ingredients via checkboxes, and the app infers the most likely cuisine type using the trained model. The frontend uses basic HTML and JavaScript to interact with the model, providing a smooth user experience.

## Conclusion and Resources
Free machine learning resources are widely available, covering topics such as clustering, regression, and natural language processing. Developers are encouraged to explore open repositories, join challenges, and contribute to community-driven projects to deepen their practical understanding of ML integration in web applications.

## Introduction to Applied Machine Learning and ONNX Runtime
Applied machine learning connects app builders (using JavaScript, Python) with data scientists, facilitating mutual understanding and collaboration between these traditionally separate disciplines. The app ecosystem has evolved; modern apps like TikTok and Instagram are infused with machine learning, making them "smart apps" that deliver personalized experiences, but they are complex for indie developers to build without ML integration.

## Challenges in Building Smart Apps
Indie developers face challenges due to the complexity of integrating ML into apps, leading many to join larger corporations to access resources and expertise. Developers must decide on app architecture (native, cross-platform, web), with the web increasingly capable of supporting intelligent, ML-infused applications due to advances in browser APIs and responsive design.

## Bridging the Skills Gap Between Web Developers and ML Engineers
Web developers focus on usability, front-end/back-end integration, and application delivery using languages like HTML, CSS, and JavaScript.ML engineers specialize in data preparation, model training, and deployment, often using Python and managing ML Ops for continuous model improvement.ONNX Runtime acts as a bridge to enable ML models to be used within web apps, easing integration despite differences in language and skill sets.

## JavaScript Ecosystem for Machine Learning
TensorFlow.js is a popular library that allows ML models to run and be trained in the browser with GPU acceleration, supporting applications like image recognition and transfer learning. ml5.js builds atop TensorFlow.js to simplify ML tasks in the browser, offering tools for image, sound, and text analysis, making ML accessible for education and experimentation. Magenta.js focuses on creative AI applications such as music and art generation, also leveraging TensorFlow.js and GPU acceleration for real-time inference in the browser. PoseNet estimates human body poses for interactive applications, exemplifying ML use in real-time video analysis, although its active development status is uncertain. Brain.js supports neural networks in JavaScript for browser and Node.js use, suitable for beginners but limited in model variety and complexity.

## Training Machine Learning Models: Browser vs. Python
Training ML models directly in the browser is possible, but it is inefficient and unsuitable for production. Robust training is typically performed in Python environments using tools such as Jupyter notebooks or .py scripts. Python offers a straightforward, high-level syntax for ML development and data handling, making it accessible even for developers primarily experienced in JavaScript. Tools like Loeb.ai provide low-code desktop solutions for training image recognition models locally without deep Python knowledge, facilitating model creation and export for web use.

##Not All ML Requires Neural Networks
Many ML problems do not require neural networks; classical ML methods remain effective and simpler for many tasks. ONNX supports classical ML frameworks like scikit-learn, enabling models built without neural networks to be deployed with ONNX Runtime in web applications.

## Case Study: Cuisine Recommendation App
A small labeled dataset from Kaggle with ingredient presence (binary features) and cuisine labels (Indian, Thai, Korean, Japanese, Chinese) was used for classification. Data preprocessing involved cleaning (dropping common, non-discriminative ingredients) and balancing to ensure equal representation of each cuisine class. Supervised multi-class classification was chosen due to labeled data and multiple cuisine categories. Logistic regression with a one-vs-rest (OVR) classifier and liblinear solver was first used, achieving ~71% accuracy. The Support Vector Classifier (SVC) with a linear kernel achieved an accuracy of ~83%, highlighting the importance of algorithm and solver choice.

## Model Conversion and Deployment with ONNX Runtime
The trained scikit-learn model was converted to ONNX format using the skl2onnx library, producing a .onnx file for runtime inference. The ONNX model can be inspected with tools like Netron to visualize input/output shapes and operators. The ONNX Runtime npm package enables importing and running the ONNX model in a web app asynchronously, performing inference based on user input from checkboxes representing ingredients. The web app suggests cuisines based on selected ingredients, serving as a practical recommendation engine example built with minimal UI and JavaScript code.


#  ONNX Model 

**ONNX (Open Neural Network Exchange)** is an open-source format designed to **represent machine learning models**. It enables **interoperability between different deep learning frameworks** such as PyTorch, TensorFlow, scikit-learn, and others.

Created by **Microsoft and Facebook**, ONNX makes it easier to move models between tools so that developers can choose the best framework for training and the best one for inference.

---

## Key Features of ONNX

- **Interoperability**: Train a model in one framework (e.g., PyTorch) and deploy it using another (e.g., TensorFlow, Caffe2, or ONNX Runtime).
- **Open Format**: ONNX is open standard and community-driven.
- **Extensibility**: Custom operators and tools can be added to ONNX.
- **Cross-Platform Support**: Supports deployment on cloud, mobile, edge, and browser environments.

---

##  ONNX Model Structure

An ONNX model consists of:

1. **Model Graph**: A directed graph that represents the computation (operators and data flow).
2. **Operators**: Standardized computations like convolution, ReLU, matmul, etc.
3. **Tensors**: Multidimensional arrays used as inputs, outputs, and intermediate variables.
4. **Initializers**: Predefined constant tensors (e.g., model weights).
5. **Metadata**: Describes the model's inputs, outputs, and version.

---



### Example: PyTorch to ONNX to Inference

```python
import torch
import torch.onnx
import torchvision.models as models

# Load a pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "resnet18.onnx", 
                  input_names=["input"], output_names=["output"])
```


# Netron in Machine Learning: A Full Explanation

**Netron** is a viewer for neural network, machine learning, and deep learning models. It provides a graphical interface that allows users to visualize model architectures and understand the structure of complex networks. Netron is widely used by data scientists, machine learning engineers, and researchers for debugging and model introspection.

## Key Features

- **Model Visualization:** Interactive visualization of layers, operations, inputs, outputs, and connections.
- **Multi-Framework Support:** Compatible with a variety of model formats including:
  - TensorFlow (`.pb`, `.tflite`, `.lite`, `.json`)
  - Keras (`.h5`, `.keras`)
  - PyTorch (`.pt`, `.pth`, `.pkl`, TorchScript)
  - ONNX (`.onnx`)
  - CoreML (`.mlmodel`)
  - Caffe (`.caffemodel`, `.prototxt`)
  - MXNet (`.model`, `.json`)
  - PaddlePaddle, Darknet, CNTK, and many others.

## Use Netron

Netron simplifies the understanding of how models are structured, especially when:
- You are working with pre-trained models.
- You need to verify the input/output shapes.
- You want to ensure layer connections are correct.
- You’re collaborating across teams with different levels of ML expertise.

## Installation and Usage

### Option 1: Use Netron Online
Simply drag and drop your model file onto the Netron web app — no installation required.

### Option 2: Install on Desktop
Netron is available for **Windows**, **macOS**, and **Linux**.

#### Using pip (for Python users):
```bash
pip install netron
netron your_model.onnx
```


# Deploying Python Machine Learning Web Apps with Voilà


**Voilà** is an open-source Python library that turns Jupyter Notebooks into interactive, code-free web applications. It's ideal for presenting **machine learning models** in a user-friendly, browser-accessible format using `ipywidgets`.

### Features

- Converts `.ipynb` notebooks into live browser apps
- Hides code cells and only shows outputs/widgets
- Integrates easily with `ipywidgets`, `scikit-learn`, `pandas`, and more
- No need for HTML, CSS, or JavaScript
- Deployable via Binder, Docker, Hugging Face Spaces, or cloud providers

---

## Installation

Use `pip` to install Voilà and supporting libraries:

```bash
pip install voila ipywidgets scikit-learn notebook
```

To lunch locally:

voila model_app.ipynb



### Sample code:

```python
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load dataset and train model
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Define input widgets
sepal_length = widgets.FloatSlider(min=4, max=8, step=0.1, description='Sepal Length')
sepal_width = widgets.FloatSlider(min=2, max=5, step=0.1, description='Sepal Width')
petal_length = widgets.FloatSlider(min=1, max=7, step=0.1, description='Petal Length')
petal_width = widgets.FloatSlider(min=0.1, max=3, step=0.1, description='Petal Width')
button = widgets.Button(description='Predict')
output = widgets.Output()

# Define interaction logic
def on_button_click(b):
    input_data = np.array([[sepal_length.value, sepal_width.value, petal_length.value, petal_width.value]])
    prediction = model.predict(input_data)
    species = iris.target_names[prediction[0]]
    with output:
        output.clear_output()
        print(f'Predicted Species: {species}')

button.on_click(on_button_click)

# Display in Voilà
display(sepal_length, sepal_width, petal_length, petal_width, button, output)

```




