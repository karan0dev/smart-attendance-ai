# Smart Attendance System: AI Face Recognition

An enterprise-grade biometric attendance system engineered with Python and Streamlit, featuring real-time 128-dimensional facial embeddings and mathematical anti-spoofing protocols.

## ⚙️ Core Architecture
* **Biometric Engine:** Replaced legacy Haar Cascades with a Deep Convolutional Neural Network (CNN) via `dlib`. Projects facial geometry into a 128-dimensional vector space, utilizing Euclidean distance for high-accuracy identification under dynamic lighting.
* **Liveness Detection (Anti-Spoofing):** Implemented a 68-point facial landmark predictor to actively calculate the Eye Aspect Ratio (EAR). The system mathematically verifies biological liveness (blinking) before authorizing an attendance log, successfully defeating photograph and digital screen spoofing attacks.
* **Hardware Optimization:** Engineered spatial downscaling algorithms to reduce mathematical processing loads by 75%, ensuring stable 30 FPS real-time webcam rendering on standard consumer CPUs without input lag.
* **Memory Management:** Enforced contiguous memory mapping (`np.ascontiguousarray`) and strict 8-bit integer casting (`uint8`) to bridge high-level OpenCV frames with the rigid C++ `dlib` backend, preventing heap corruption.

## 🛠️ Tech Stack
Python 3.12 | OpenCV | face_recognition | Streamlit | Pandas | NumPy 1.26.4
