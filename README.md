### **README.md**
# Canny Edge Detector - NI-GPU

## **📌 Project Stages**
The project is structured into the following **four stages**, with each stage contained in a separate directory (`stage-n/`):

### **🔹 Stage 1: Sequential Implementation**
- Implement a **basic C++ version** of the Canny Edge Detector.
- Uses **OpenCV** for image I/O but implements edge detection manually.
- Serves as a baseline for further optimizations.

### **🔹 Stage 2: Parallel Implementation (GPU)**
- Convert the algorithm to run on **CUDA** for parallel execution.
- Optimize performance by leveraging **GPU processing power**.
- Compare execution time against the sequential version.

### **🔹 Stage 3: Benchmarking**
- Measure the performance of **both sequential and parallel implementations**.
- Evaluate speedup factors, memory usage, and compute efficiency.
- Perform tests on **different image sizes** and **varying kernel sizes**.

### **🔹 Stage 4: Final Report & Results**
- Summarize findings in a **detailed report**.
- Compare **accuracy, performance, and efficiency**.
- Discuss challenges faced and potential future improvements.

---

## **💾 Project Structure**
```
/CannyEdgeDetector
 ├── stage-1/       # Sequential CPU implementation
 ├── stage-2/       # Parallel GPU (CUDA) implementation
 ├── stage-3/       # Benchmarking and performance evaluation
 ├── stage-4/       # Final report and results
 ├── data/          # Sample images for testing
 ├── README.md
 ├── .gitignore
```

---

## **🔧 Dependencies**

- **C++17 or later** (`g++`, `clang`, or MSVC)
- **OpenCV** (`libopencv-dev`)
- **CUDA Toolkit** (for Stage 2 and beyond)