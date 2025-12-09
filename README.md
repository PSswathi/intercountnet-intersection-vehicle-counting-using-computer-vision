
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

YOLO training and validation runs are in,

../intercountnet-intersection-vehicle-counting-using-computer-vision/blob/main/yolo_train_runs

../intercountnet-intersection-vehicle-counting-using-computer-vision/blob/main/yolo_val_runs

Also to visualize test results for any random images results are saved in 

../intercountnet-intersection-vehicle-counting-using-computer-vision/blob/main/test_dataset_vehicle_counts.json

DETR results are stored in,

../intercountnet-intersection-vehicle-counting-using-computer-vision/blob/main/detr_results


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

6. Deployment 
*	FastAPI or Flask (REST API for inference)
*	Docker (containerization)
*	AWS EC2 / Lambda / S3 (cloud hosting)

7. Version Control & Workflow
*	Git & GitHub

## Setup

1. Create conda environment
```bash
# Frontend
cd streamlit_app
conda env create -f environment.yml

cd ..
#Backend
conda env create -f environment.yml
```
2. Activate the environment
```bash
conda activate vehicle_detection
```
3. Run the app
```bash
cd streamlit_app
streamlit run app_yolo.py
```
4. Update conda environment
```bash
cd streamlit_app
conda env update -n vehicle_detection -f environment.yml
```

## Deploy the Stack + Model

### Prerequisites
- AWS CLI installed and configured (`aws configure`)
- Key-pair `xx-key` created in the target region
- Git repo cloned locally (contains `aws/intercount-aws-infra.yaml` & `deploy.sh`)

---

### 1. Upload Model to S3

Stores the model artifact where the EC2 instance (via IAM role) can download it at start-up.

```bash
# one-time: create bucket (if not exists)
aws s3 mb s3://intercountnet-models --region us-east-1

# copy your trained weights
aws s3 cp models/best.pt s3://intercountnet-models/best-yolov8n.pt
```

### 2. Launch Infrastructure

Performs below:
1. Uploads the CloudFormation template
2. Creates security group, IAM role, t3.micro EC2
3. Clones the repo, builds Docker image, starts container on port 80
4. waits until healthy and outputs the public URL

```bash
chmod +x ./aws/deploy.sh
./aws/deploy.sh intercountnet us-east-1
```

```
Stack created successfully.
[
  {
    "OutputKey": "PublicIp",
    "OutputValue": "34.226.233.101"
  },
  {
    "OutputKey": "StreamlitURL",
    "OutputValue": http://34.226.233.101"
  }
]
```
Open the Streamlit URL 'http://34.226.233.101' in a browser – the app should load and download the model on first inference.

### 3. Update Infrastructure

```bash
chmod +x ./aws/update.sh
./aws/update.sh intercountnet us-east-1
```

### 4. Tear Down When Done

Deletes the entire stack (EC2, security group, IAM role) and stops any charges.

```bash
chmod +x ./aws/delete.sh
./aws/delete.sh intercountnet us-east-1
```

## Sample Prediction image:

![Alt Text](figs/demo.png)


## Project deliverables shared links:

https://drive.google.com/drive/folders/1Y9yDMsb0EPvTQBobA3s9zI-IrSJEKIJO?usp=sharing

Video Presentations:

https://drive.google.com/file/d/1Ec_qwv2iO_MUFOdAZtAziKsD2NM8GURu/view?usp=sharing

Inferences Recorded:

https://drive.google.com/drive/folders/1VAET8f3ZWHwLT8y0nSEfZSEhOlukpfP9?usp=sharing

For sample video example for testing the inference ,

https://drive.google.com/file/d/1adX73S3BmQkVz_VHIv-eQxoieKVxQ5p4/view?usp=sharing


## References:

Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal speed and accuracy of object detection. arXiv. https://arxiv.org/abs/2004.10934

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. In European Conference on Computer Vision (ECCV). https://arxiv.org/abs/2005.12872

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV). https://arxiv.org/abs/1708.02002

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/1912.01703

Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv. https://arxiv.org/abs/1804.02767

Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in Neural Information Processing Systems (NIPS). https://arxiv.org/abs/1506.01497

Ultralytics. (2023). Ultralytics YOLOv8 documentation. Ultralytics. https://docs.ultralytics.com/

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/1706.03762

Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv. https://arxiv.org/abs/2207.02696

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., … & Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI). https://www.tensorflow.org/

TensorFlow. (2023). TensorFlow API documentation. TensorFlow. https://www.tensorflow.org/api_docs

PyTorch. (2023). PyTorch documentation. PyTorch. https://pytorch.org/docs/
