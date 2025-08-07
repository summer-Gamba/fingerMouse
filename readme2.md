<h1 align="center">Vision-Based HCI System</h1>
<p align="center">
  A smart interface to control a Raspberry Pi 5 using real-time face recognition and hand gesture analysis.
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi%205-orange.svg?logo=raspberrypi" alt="Platform">
</p>

<p align="center">
  <img src="link_to_your_project_demo.gif" alt="Project Demo">
</p>

## About The Project

이 프로젝트는 키보드나 마우스와 같은 전통적인 입력 장치 없이, 컴퓨터 비전 기술을 활용하여 사용자와 컴퓨터 간의 상호작용(HCI)을 구현하는 것을 목표로 합니다. 라즈베리파이 5와 카메라 모듈을 사용하여, 얼굴 인식을 통한 사용자 인증 및 손 제스처를 통한 마우스 제어, 화면 캡처, OCR 등의 복합적인 작업을 수행하는 통합 시스템입니다.

---

## Key Features

* **Multi-Angle Face Recognition Login:** 사용자의 정면, 좌측, 우측 얼굴 데이터를 등록하여 인식률을 향상시켰으며, 3초간의 지속적인 인식을 통해 안정적인 로그인 세션을 제공합니다.
* **Gesture-Based System Control:** MediaPipe를 활용한 실시간 손 추적을 통해 두 가지 주요 제어 모드를 지원합니다.
    * **Mouse Control Mode:** `Pointer` 제스처로 시스템 마우스 커서를 제어합니다.
    * **Screen Capture & OCR Mode:** `Open` 제스처로 모드를 전환하고, 마우스 드래그를 통해 선택한 화면 영역을 캡처하여 OCR을 실행합니다.
* **Robust Screen Capture UI:** 시스템 호환성 문제를 해결하기 위해, 캡처 시 화면 전체의 스크린샷을 **OpenCV 창에 직접 렌더링**하고 그 위에 마우스로 영역을 선택하는 방식을 채택하여 UI 안정성을 확보했습니다.
* **OCR Result Visualization:** 화면 캡처 후, 추출된 텍스트는 터미널에 출력될 뿐만 아니라, 인식된 위치에 경계 상자와 텍스트가 그려진 별도의 결과 창을 통해 시각적으로 제공됩니다.

---

## System Workflow

1.  **Application Start:** 사용자가 Tkinter 기반의 메인 GUI를 실행합니다.
2.  **Face Recognition Login:** `Start` 버튼을 누르면 카메라가 활성화되고, `FaceRecognitionManager`가 사용자 인증을 시작합니다.
3.  **Hand Tracking Activation:** 로그인이 성공하면, 시스템은 `HandTrackingManager`를 초기화하고 '손 추적 모드'로 자동 전환됩니다.
4.  **Function Execution:** 사용자는 제스처를 통해 '마우스 제어' 또는 '화면 캡처' 기능을 선택하여 작업을 수행합니다.
5.  **Capture & OCR:** '화면 캡처' 모드에서 영역을 선택하고 확정하면, 해당 이미지가 저장되고 즉시 OCR 파이프라인으로 전달됩니다.
6.  **Display Results:** OCR 결과는 터미널과 별도의 시각화 창에 동시에 출력됩니다.

---

## Built With

| Category      | Technology / Library                                                              |
| :------------ | :-------------------------------------------------------------------------------- |
| **Hardware** | Raspberry Pi 5, Raspberry Pi Camera Module 3                                  |
| **OS** | Raspberry Pi OS (Debian based, X11 Desktop Environment)                                                  |
| **Core** | Python 3, OpenCV, Tkinter, Threading, Queue                                     |
| **Input/Cam** | Pynput, PyAutoGUI, Picamera2                                                |
| **AI / ML** | MediaPipe, TFLite Runtime, TensorFlow, NumPy             |
| **AI Models** | **Face:** Lightweight-Face-Detection, MobileFaceNet<br>**Hand:** MediaPipe Hands, Custom KeyPointClassifier<br>**OCR:** EAST, CRNN-based Recognizer |

---

## Getting Started

### Prerequisites
* Raspberry Pi 5
* Raspberry Pi Camera Module 3

### Installation Guide

1.  **Clone the repository.**
    ```bash
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    cd your_repository
    ```

2.  **Verify the file structure.**
    프로젝트가 올바르게 작동하려면 아래의 파일 구조를 반드시 유지해야 합니다.
    ```
    /Your_Project_Folder/
    │
    ├── main.py
    │
    ├── frozen_east_text_detection.pb
    ├── recognizer_model.tflite
    │
    ├── face/
    │   ├── Lightweight-Face-Detection.tflite
    │   └── MobileFaceNet_9925_9680.tflite
    │
    └── handMini2/
        ├── model/
        │   └── keypoint_classifier/
        │       ├── keypoint_classifier.py
        │       └── keypoint_classifier.tflite
        └── utils/
            └── calc_landmark.py
    ```

3.  **Install all dependencies.**
    터미널을 열고 아래 명령어를 순서대로 실행하여 모든 환경 설정 및 라이브러리를 설치합니다.
    ```bash
    # Step 1: Install system-level dependencies
    sudo apt-get update
    sudo apt-get install -y python3-tk python3-pil.imagetk

    # Step 2: Install required Python packages
    pip install opencv-python numpy mediapipe pynput picamera2 pyautogui tflite-runtime tensorflow
    ```

4.  **Configure Raspberry Pi.**
    * **Enable Camera:** `sudo raspi-config` → `3 Interface Options` → `I1 Legacy Camera` → **Enable**.
    * **Use X11:** `sudo raspi-config` → `6 Advanced Options` → `A6 Wayland` → **W1 X11**. (필수: `pyautogui` 호환성)
    * **Reboot** the Raspberry Pi after making changes.

### Running the Application
프로젝트 폴더로 이동 후, 아래 명령어를 실행합니다.

```bash
python main.py
```

---

## Usage

1.  **Login:** 프로그램 실행 후 `Start Face Recognition` 버튼을 클릭하고 카메라에 얼굴을 3초간 인식시켜 로그인합니다.
2.  **Hand Tracking:** 로그인에 성공하면 손 추적 모드가 자동으로 시작됩니다. `검지` 제스처로 마우스를 제어할 수 있습니다.
3.  **Mode Switch:** `손바닥` 제스처를 취하면 `Screen Capture & OCR` 모드로 전환됩니다.
4.  **Screen Capture:** 캡처 모드에서는 **실제 마우스**를 사용하여 OpenCV 창에 표시된 스크린샷 위에서 원하는 영역을 클릭 & 드래그합니다.
5.  **Confirm & OCR:** 영역 선택 후 키보드에서 **`c`** 키를 누르면 캡처가 확정되고, 잠시 후 OCR 결과가 새 창에 나타납니다.
6.  **Quit:** `q` 키를 누르면 프로그램이 종료됩니다.

---

## Key Technical Challenges & Solutions

* **Model Porting for Embedded Systems:**
    * **Challenge:** 초기 모델 후보였던 FaceNet, ArcFace는 TFLite에서 지원하지 않는 연산을 포함하여 임베디드 배포용으로 변환하는 데 실패했습니다.
    * **Solution:** 프로젝트 전략을 수정하여, 모바일 환경에 최적화된 `MobileFaceNet`과 `Lightweight-Face-Detection` 모델을 채택하여 TFLite 호환성 및 배포 용이성을 확보했습니다.

* **Resolving Library Conflicts:**
    * **Challenge:** MediaPipe, OpenCV DNN, TFLite 등 여러 AI 라이브러리를 동시에 초기화할 때 내부 라이브러리 충돌로 인해 손 추적 기능이 간헐적으로 실패하는 문제가 발생했습니다.
    * **Solution:** **'지연 로딩(Lazy Loading)'** 기법을 도입했습니다. 로그인 직후에는 손 추적에 필수적인 모델만 로드하고, 사용자가 실제로 캡처/OCR 기능을 사용하는 제스처를 취했을 때 비로소 무거운 OCR 모델을 로드하여 충돌을 방지하고 초기 반응 속도를 개선했습니다.

* **Robust UI Implementation:**
    * **Challenge:** Tkinter의 투명 오버레이 창을 이용한 캡처 UI는 라즈베리파이 데스크톱 환경과의 호환성 문제로 인해 렌더링 오류가 반복되었습니다.
    * **Solution:** UI 구현 방식을 완전히 변경하여, **OpenCV 창에 직접 스크린샷을 배경으로 띄우고 `cv2.setMouseCallback`을 이용해 마우스 이벤트를 처리**하는 방식으로 안정성을 확보했습니다.

---

## Future Work

-   캡처된 텍스트의 클립보드 복사 기능 추가
-   GPU/NPU 가속을 활용한 AI 모델 추론 성능 최적화
-   특정 제스처(예: 'OK' 사인)에 커스텀 매크로 기능 할당 기능 개발
