# 얼굴 인식 및 손 제스처 기반 스마트 인터페이스 시스템

**라즈베리파이 5와 카메라 모듈을 활용하여, 사용자의 얼굴과 손 제스처만으로 PC를 제어하는 차세대 HCI(Human-Computer Interaction) 프로젝트입니다.**

---

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 별도의 키보드나 마우스 없이, 사용자의 생체 정보(얼굴)와 행동(손 제스처)만으로 컴퓨터 시스템에 접근하고 다양한 작업을 수행하는 것을 목표로 합니다. 얼굴 인식을 통해 사용자를 인증하는 로그인 시스템을 구축하고, 로그인 후에는 손 추적 기술을 통해 마우스 제어, 화면 캡처, 그리고 캡처된 이미지 내의 텍스트를 인식하는 OCR 기능까지 연동하여 통합적인 스마트 인터페이스를 구현했습니다.

## 2. 주요 기능 (Key Features)

* **👤 다중 각도 얼굴 인식 로그인:** 한 사람당 여러 각도의 얼굴 특징(임베딩)을 데이터베이스에 저장하여, 다양한 각도와 환경에서도 높은 정확도로 사용자를 인증합니다. 3초간 지속적인 인식을 통해 보안성을 강화했습니다.
* **🖐️ 실시간 손 제스처 제어:** 로그인 성공 시, MediaPipe를 활용한 손 추적 기능이 자동으로 활성화됩니다.
    * **마우스 컨트롤:** '검지' 제스처를 통해 시스템의 마우스 커서를 실시간으로 정밀하게 제어합니다.
    * **모드 전환:** '손바닥' 제스처를 통해 '마우스 제어 모드'와 '화면 캡처 모드'를 전환합니다.
* **🖥️ 안정적인 화면 캡처 UI:** 시스템 호환성 문제를 원천적으로 해결하기 위해, 불안정한 투명 오버레이 대신 **OpenCV 창에 직접 스크린샷을 띄우고 그 위에 마우스로 영역을 선택**하는 가장 안정적인 방식을 채택했습니다. 드래그하는 동안 실시간으로 선택 영역이 표시됩니다.
* **📄 캡처 후 자동 OCR 실행:** 사용자가 화면의 특정 영역을 캡처하면, 해당 이미지는 파일로 저장되는 동시에 **자동으로 OCR(광학 문자 인식) 기능이 실행**됩니다.
* **📊 OCR 결과 시각화:** 터미널에 텍스트를 출력하는 것을 넘어, 캡처된 원본 이미지 위에 인식된 글자의 위치마다 **경계 상자와 인식 결과가 그려진 별도의 결과 창**을 시각적으로 제공합니다.

## 3. 시스템 아키텍처 및 작동 흐름 (Architecture & Workflow)

본 시스템은 다중 스레드(Multi-threaded) 아키텍처를 기반으로, 무거운 AI 연산과 GUI 업데이트가 서로를 방해하지 않고 부드럽게 작동하도록 설계되었습니다.

1.  **시작:** 사용자가 Tkinter로 제작된 메인 GUI 창에서 '얼굴 인식 시작' 버튼을 누릅니다.
2.  **로그인:** 별도의 스레드에서 카메라가 활성화되고, `FaceRecognitionManager`가 실시간으로 얼굴을 탐지 및 인식합니다. 등록된 사용자가 3초간 인식되면 로그인에 성공합니다.
3.  **모드 전환:** 로그인 성공 시, 시스템은 자동으로 `HandTrackingManager`를 초기화하며 '손 추적 모드'로 전환됩니다.
4.  **기능 수행:**
    * **마우스 제어:** 'Pointer' 제스처로 마우스를 제어합니다.
    * **캡처 모드 진입:** 'Open' 제스처로 '화면 캡처 모드'로 전환하면, 메인 OpenCV 창의 화면이 실시간 카메라 영상에서 전체 데스크톱 화면의 스크린샷으로 바뀝니다.
    * **영역 선택 및 OCR:** 사용자는 마우스로 스크린샷 위에서 원하는 영역을 드래그하여 선택하고, 키보드 `c`키로 확정합니다. 확정된 영역은 즉시 이미지 파일로 저장되고, 동시에 OCR 기능이 실행됩니다.
5.  **결과 확인:** OCR 결과가 시각화된 별도의 창이 화면에 나타납니다. 사용자는 결과를 확인한 후 창을 닫고, `Open` 제스처로 다시 '마우스 제어 모드'로 돌아올 수 있습니다.

## 4. 기술 스택 (Tech Stack)

* **하드웨어 (Hardware):**
    * Raspberry Pi 5
    * Raspberry Pi Camera Module 3
* **운영체제 (OS):**
    * Raspberry Pi OS (Debian based)
* **핵심 라이브러리 (Core Libraries):**
    * **AI / Computer Vision:** OpenCV, MediaPipe, TFLite Runtime, TensorFlow, NumPy
    * **GUI & Input Control:** Tkinter, Pynput, PyAutoGUI
    * **Camera Control:** Picamera2
* **사용한 AI 모델 (AI Models):**
    * **Face Detection:** `Lightweight-Face-Detection` (TFLite)
    * **Face Recognition:** `MobileFaceNet` (TFLite)
    * **Hand Landmark:** `MediaPipe Hands`
    * **Gesture Classification:** Custom `KeyPointClassifier` (TFLite)
    * **Text Detection:** `EAST (Efficient and Accurate Scene Text Detector)` (OpenCV DNN)
    * **Text Recognition:** CRNN-based `recognizer_model` (TFLite)

## 5. 설치 및 실행 방법 (Installation & Usage)

#### 5.1. 파일 구조
프로젝트를 실행하기 위해, 아래와 같은 폴더 및 파일 구조를 유지해야 합니다.

```
/Your_Project_Folder/
│
├── main.py  (메인 실행 스크립트)
│
├── frozen_east_text_detection.pb  (OCR 모델)
├── recognizer_model.tflite        (OCR 모델)
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

#### 5.2. 라이브러리 설치
터미널을 열고 아래 명령어를 순서대로 실행하여 필요한 모든 라이브러리를 설치합니다.

```bash
# 1. 시스템 라이브러리 업데이트 및 설치
sudo apt-get update
sudo apt-get install -y python3-tk python3-pil.imagetk

# 2. 필수 파이썬 라이브러리 설치
pip install opencv-python numpy mediapipe pynput picamera2 pyautogui tflite-runtime tensorflow
```

#### 5.3. 라즈베리파이 설정
1.  **카메라 활성화:** 터미널에 `sudo raspi-config` 입력 → `3 Interface Options` → `I1 Legacy Camera` → **Enable** 선택 후 재부팅.
2.  **디스플레이 서버 변경:** 터미널에 `sudo raspi-config` 입력 → `6 Advanced Options` → `A6 Wayland` → **W1 X11** 선택 후 재부팅. (화면 캡처 기능의 안정성을 위해 필요)

#### 5.4. 프로그램 실행
프로젝트 폴더로 이동한 후, 터미널에 다음 명령어를 입력하여 프로그램을 실행합니다.

```bash
python main.py
```

## 6. 주요 개발 과정 및 문제 해결 (Challenges & Lessons Learned)

프로젝트를 진행하며 여러 기술적 난관에 부딪혔으며, 이를 해결하는 과정에서 많은 것을 배울 수 있었습니다.

* **도전 과제 1: AI 모델의 TFLite 변환 실패 및 전략 수정**
    * **문제:** 초기 후보였던 FaceNet, ArcFace 모델이 TFLite에서 지원하지 않는 연산을 포함하여 라즈베리파이 배포용으로 변환하는 데 실패했습니다.
    * **해결:** 전략을 수정하여, 처음부터 모바일 환경에 최적화되고 TFLite 변환이 용이한 MobileFaceNet과 Lightweight-Face-Detection 모델을 재탐색하고 성공적으로 적용했습니다. 이는 **프로젝트 초기에 타겟 환경에서의 실행 가능성을 최우선으로 검증하는 것**의 중요성을 깨닫게 했습니다.

* **도전 과제 2: 다중 AI 모델 간의 라이브러리 충돌**
    * **문제:** 얼굴 인식(TFLite), 손 인식(MediaPipe), OCR(OpenCV DNN) 등 여러 AI 라이브러리를 동시에 초기화할 때, 내부 라이브러리 경로가 꼬이면서 MediaPipe의 손 인식 기능이 원인 모르게 중단되는 현상이 발생했습니다.
    * **해결:** **'지연 로딩(Lazy Loading)'** 기법을 도입했습니다. 로그인 직후에는 손 추적에 필요한 최소한의 모델만 로드하고, 사용자가 실제로 캡처/OCR 기능을 사용하는 제스처를 취했을 때 비로소 무거운 OCR 모델을 로드하도록 하여 라이브러리 간의 충돌을 원천적으로 방지하고 초기 반응 속도를 개선했습니다.

* **도전 과제 3: 화면 캡처 UI의 불안정성**
    * **문제:** Tkinter의 투명 오버레이 창을 이용한 초기 캡처 UI는 라즈베리파이의 데스크톱 환경과 호환성 문제를 일으켜, 화면이 회색으로 변하거나 선택 영역이 보이지 않는 등 치명적인 버그가 반복되었습니다.
    * **해결:** 접근 방식을 완전히 변경하여, 불안정한 Tkinter 오버레이 대신 **안정성이 보장된 OpenCV 창에 직접 스크린샷을 배경으로 띄우고, `cv2.setMouseCallback`을 이용해 마우스 이벤트를 처리**하는 방식으로 전면 재설계했습니다. 이로써 모든 환경에서 일관되고 안정적인 캡처 UI를 구현할 수 있었습니다.

## 7. 향후 개선 과제 (Future Improvements)

* **기능 확장:** 캡처된 텍스트를 클립보드에 복사하거나, 특정 제스처(예: 'OK' 사인)에 새로운 매크로 기능을 할당.
* **성능 최적화:** 현재 CPU 기반으로 동작하는 AI 모델들을 라즈베리파이 5의 GPU/NPU 가속을 활용하도록 최적화.
