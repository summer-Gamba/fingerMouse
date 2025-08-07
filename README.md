# ➤ fingerMouse: Face Recognition & Hand Tracking & OCR System for Raspberry Pi 5

> 얼굴 인식 + 손 제스처 기반 마우스 제어 + OCR 텍스트 인식 시스템  

<br>

## 시스템 구조
```bash
얼굴 인식 → 사용자 로그인 (첫 등록은 r입력)
        ↓
손 제스처 인식
    ├─ Pointer : 마우스 제어
    ├─ Open    : 모드 전환
    └─ Close   : 캡처 실행
        ↓
화면 캡처 + OCR 수행
```
<br>

### 주요 기능

| 기능 | 설명 |
|------|------|
|  얼굴 인식 로그인 | TFLite 기반 Lightweight Face Detection + MobileFaceNet으로 본인 인증 |
|  손 제스처 인식 | MediaPipe Hands + MLP 기반 제스처 분류 (Open / Pointer / Close 등) |
|  제스처 마우스 이동 & 클릭 | 'Pointer' 제스처로 마우스 이동, 손가락 고정으로 클릭 |
|  OCR 화면 캡처 모드 | 'Open' 제스처로 모드 전환 후 손가락으로 사각형을 그려 영역 지정 |
|  TFLite OCR 텍스트 인식 | EAST + CTC 기반 문자 인식 모델로 선택 영역의 텍스트 추출 |
|  오버레이 시각화 | Tkinter 기반 전체화면 오버레이에 실시간 박스 시각화 |


<br>

## 실행 방법

```bash
python3 main.py
```
<br>

### 제스처 제어 방식

| 제스처 / 키보드         | 기능                                      |
|-------------------------|-------------------------------------------|
| `Open` 🖐️                | 모드 전환 (마우스 제어 ↔ 캡처 & OCR)     |
| `Pointer` ☝️           | 커서 이동 또는 캡처 영역 지정             |
| `Pointer (1.5초 고정)` | 클릭 또는 영역 시작/끝 지정              |
| `Close` ✊              | 선택한 영역을 캡처하고 OCR 수행           |
| `r` 키                 | 캡처 리셋                                 |
| `q` 키                 | 시스템 종료                               |




