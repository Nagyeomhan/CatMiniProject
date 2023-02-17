# 반려묘 행동 분석 프로젝트

1. 프로젝트 목적 : 영상 업로드 시 반려묘의 행동을 분석해주는 시스템 구현
2. 프로젝트 기간 : 약 10일
3. 프로젝트 멤버 : 전처리 및 초기모델 1명, 후기모델 1명
4. 사용 데이터 : <a href='https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=59'>[AI-Hub] 반려동물 구분을 위한 동물 영상</a>
5. 개발 환경<br>
(1) 주요 개발 언어 : Python<br>
(2) 모델링 및 학습 ① : Seagate 1TB HDD 외장하드 / GeForce RTX 1050 Ti / CUDA 11.7 ver<br>
(3) 모델링 및 학습 ② : Google Colab GPU<br>
<br>

## 사용 모델

**1. Keypoint R-CNN**
<br>
Mask R-CNN 기반으로 이미지에 관절 포인트를 매칭하여 학습하는데 활용

**2. HRNet**
<br>
관절 포인트별로 confidence score를 도출하는데 활용

**3. ST-GCN**
<br>
관절 포인트를 기반으로 동작을 예측하는데 활용
<br>
<br>
  
## 발표 자료 요약
![화면 캡처 2022-09-27 101159](https://user-images.githubusercontent.com/108378151/192411049-9bca0a3b-7f02-4bf1-8099-dff296001e7c.png)
![화면 캡처 2022-09-27 101244](https://user-images.githubusercontent.com/108378151/192411057-e206fe02-b660-423a-ab92-79fdda28f978.png)
![화면 캡처 2022-09-27 101300](https://user-images.githubusercontent.com/108378151/192411064-27ecc1ac-afc0-43a4-8a0c-b98b06ad54ff.png)
![화면 캡처 2022-09-27 101313](https://user-images.githubusercontent.com/108378151/192411069-26f23597-a5b8-4f08-ad85-83f792048b94.png)
![화면 캡처 2022-09-27 101343](https://user-images.githubusercontent.com/108378151/192411078-13c3eca5-69d5-433b-936e-06375b938058.png)
![화면 캡처 2022-09-27 101404](https://user-images.githubusercontent.com/108378151/192411090-5d00cfa7-f124-4f9c-a5eb-fe7e1fb4a30d.png)
![화면 캡처 2022-09-27 101418](https://user-images.githubusercontent.com/108378151/192411097-8b681bb5-5802-4aa8-bab6-c145fca4bab9.png)
