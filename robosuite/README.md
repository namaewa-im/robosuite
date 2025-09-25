## Quick Guide (Minimal)

- 위치: `robosuite/environments/manipulation/`
- 실행 전: 의존성 설치, GUI 렌더러 필요

```bash
cd /home/tree/robosuite
pip install -r requirements.txt
```

### 1) Keyboard Teleop
- 파일: `pick_place_teleop.py`
- 한 줄: 키보드로 Panda 조작, EE `EE[x,y,z]`, `EE(rpy°)[r,p,y]`와 오브젝트 좌표 출력
- 실행:
```bash
python robosuite/environments/manipulation/pick_place_teleop.py
```
[Screencast from 09-25-2025 03:54:51 PM.webm](https://github.com/user-attachments/assets/bd16e6b2-ac4a-44fa-bbd7-12d68b486d90)
[Screencast from 09-25-2025 03:57:54 PM.webm](https://github.com/user-attachments/assets/9f0aeaba-a902-4cf4-b918-aac57536e6d9)


#### view 전환 키 -> [
[Screencast from 09-25-2025 04_03_07 PM.webm](https://github.com/user-attachments/assets/419739cf-3426-47c8-bf53-73a8ece34931)


### 2) print object xyz, eular rpy
- 파일: `pick_place_print_bread_orientation.py`
- 한 줄: 물체와 EE의 xyz 출력
- 실행:
```bash
python robosuite/environments/manipulation/pick_place_print_bread_orientation.py
```
<img width="1302" height="997" alt="image" src="https://github.com/user-attachments/assets/171a3c91-dc8a-4c35-980b-c2690b726417" />
<img width="1302" height="997" alt="image" src="https://github.com/user-attachments/assets/6b4e2c5b-3246-4288-ae0a-653879190757" />

### 3) Bread Lift Sequence
- 파일: `pick_place_print_bread_lifting.py`
- 한 줄: 빵 위치로 XY 정렬 → Z 하강 → 집기 → Z 상승
- 실행:
```bash
python robosuite/environments/manipulation/pick_place_print_bread_lifting.py
```
[Screencast from 09-25-2025 08:09:08 PM.webm](https://github.com/user-attachments/assets/5836dcfa-afb2-4011-87d9-84721b819165)

Tips
- 종료: Ctrl+C
- 키 입력 포커스가 뷰어에 있어야 합니다.
