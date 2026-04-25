♻️ Smart Waste Classification using AI (Deployed Web App)
🚀 Overview

This project is an AI-powered waste classification web application that has been trained using deep learning and deployed as a web app. Users can upload an image of waste, and the model predicts its correct category instantly.

The goal is to promote smart waste segregation and environmental sustainability using Artificial Intelligence.

💡 Problem Statement

Manual waste sorting is inefficient and error-prone, leading to poor recycling and environmental pollution.
This project automates waste classification using AI to improve accuracy and efficiency.

🧠 Solution

We developed a deep learning-based image classification model using transfer learning (MobileNetV2). The model analyzes waste images and predicts their category in real-time through a deployed web interface.

🗂️ Waste Categories
Plastic ♻️
Metal 🔩
Glass 🍾
Paper 📄
Cardboard 📦
Trash 🗑️

⚙️ Tech Stack
Python 🐍
TensorFlow / Keras 🤖
Streamlit 🌐
NumPy
Pillow (PIL)

🧠 Model Details
Architecture: MobileNetV2 (Transfer Learning)
Input Size: 224 × 224 images
Activation: Softmax
Loss Function: Sparse Categorical Crossentropy
Output: Waste category + confidence score

🌐 Deployment
The model is deployed as a Streamlit web application.

👉 Users can:
Upload an image
Get instant prediction
View confidence score
See recycling suggestions

The app works on both desktop and mobile browsers.

Architecture 
<img width="1024" height="455" alt="image" src="https://github.com/user-attachments/assets/8a0b3afc-84dd-465a-97f7-30b7377aff41" />


📸 Features
📤 Image upload support
🤖 Real-time AI prediction
📊 Confidence score display
♻️ Waste handling suggestions
🔐 Simple login system
📱 Mobile-friendly interface

🖥️ How to Run Locally
1. Clone repository
git clone https://github.com/saniyashariff61281-sys/SMART-WASTE-CLASSIFIER.git
2. Install dependencies
pip install -r requirements.txt
3. Run app
streamlit run app.py
📦 requirements.txt
tensorflow
streamlit
numpy
pillow

📱 Accessibility
Works on all modern browsers
Mobile-friendly web interface
Can be saved to home screen for app-like experience

🎯 Future Improvements
Multi-object detection in images
Camera-based live classification
Smart waste bin IoT integration
Improved dataset and accuracy tuning

🏆 Use Cases
Smart city waste management
Recycling automation
Environmental awareness systems
AI/ML academic and hackathon projects

👨‍💻 Project Status
✔ Model trained
✔ Web app deployed
✔ Fully functional AI system

Live Deployment
https://smart-waste-classifi-9pwq.bolt.host

Output
<img width="1599" height="899" alt="image" src="https://github.com/user-attachments/assets/81804746-6874-49a2-a21a-589791477650" />

<img width="1599" height="899" alt="image" src="https://github.com/user-attachments/assets/027c47c6-bd1e-4130-99bb-88859eb12961" />

<img width="1599" height="899" alt="WhatsApp Image 2026-04-25 at 01 07 41" src="https://github.com/user-attachments/assets/425bc154-91af-42bf-b39c-c70257330303" />

