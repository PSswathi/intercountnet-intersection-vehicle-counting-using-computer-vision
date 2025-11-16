
# InterCountNet – Intersection Vehicle Counting Network


## **Business Problem:**
Urban intersections are becoming increasingly congested as cities grow and vehicle usage rises. Traditional traffic monitoring methods—such as manual counting, fixed induction loops, or costly radar systems are either inaccurate, expensive to maintain, or incapable of capturing real-time multi-class traffic patterns.

City traffic departments lack accurate per-vehicle-class counts (cars, trucks, buses, motorcycles, bicycles) at intersections, which limits their ability to:

*	optimize signal timings
*	reduce congestion
*	assess road capacity
*	plan infrastructure upgrades
*	improve public safety
*	analyze traffic flow trends
*	enforce traffic rules efficiently

Without reliable, automatic traffic analytics, cities rely on rough estimates, leading to poor traffic planning decisions, wasted budgets, and increased accidents.

## **Project Objective:**

Develop a computer vision–based deep learning system that automatically:

*	Detects vehicles in intersection images
*	Classifies them into categories (car, truck, bus, bike, etc.)
*	Counts the number of vehicles per class in each image
*	Outputs a structured analytics report for traffic engineers

InterCountNet aims to replace manual counting and sensor-based systems with an efficient, camera-only solution.

InterCountNet is an AI-powered computer vision system that detects and counts multiple classes of vehicles at intersections using deep learning. The system enhances traffic analytics, supports smart city infrastructure planning, reduces congestion, and automates traditional manual traffic studies. It processes 17,000+ intersection images to deliver accurate, scalable, and cost-effective vehicle class counts, enabling real-time data-driven decisions for transportation authorities.


Dataset url: https://universe.roboflow.com/machine-learning-class-eiri5/intersection-traffic-piimy/dataset/10

The project will train and evaluate three state-of-the-art object detection models—YOLOv8, Faster R-CNN, and RetinaNet—on a dataset of 17,000 intersection traffic images. After preprocessing and annotation validation, each model will be trained to detect and classify vehicles (car, truck, bus, motorcycle, bicycle). The predicted bounding boxes will then be post-processed to generate per-class vehicle counts for each image. Finally, the models will be compared based on accuracy (mAP), precision/recall, inference speed, and robustness to occlusion, enabling selection of the best-performing approach for an automated intersection traffic counting system.


## **Tech stack:**

1. Programming Language
*	Python 3.10+

2. Deep Learning Frameworks
*	PyTorch (primary framework for model training)
*	Ultralytics YOLOv8 (built on PyTorch)
*	TorchVision (for Faster R-CNN & RetinaNet models)

3. Computer Vision & Data Tools
*	OpenCV (image preprocessing, visualization)
*	Albumentations (data augmentation)
*	NumPy / Pandas (data handling)

4. Model Training & Experimentation
*	Jupyter Notebook / Google Colab / AWS Sagemaker
*	Weights & Biases (W&B) or TensorBoard (experiment tracking)
*	CUDA/cuDNN for GPU acceleration

5. Model Evaluation & Reporting
*	scikit-learn (metrics, analysis)
*	Matplotlib / Seaborn / Plotly (visualization)

6. Deployment (Optional)
*	FastAPI or Flask (REST API for inference)
*	Docker (containerization)
*	AWS EC2 / Lambda / S3 (cloud hosting)

7. Version Control & Workflow
*	Git & GitHub





