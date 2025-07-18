import os
import csv
import cv2
import numpy as np
import rospkg
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def load_driving_log(log_path):
    """
    driving_log.csv 파일에서 데이터를 읽어 리스트로 반환합니다.
    """
    lines = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # 헤더는 건너뜁니다.
        for line in reader:
            # 여기에서 right_cam_path, left_cam_path로 헤더를 수정했다면 인덱스를 맞게 조절해야 합니다.
            # 지금은 cam1_path(오른쪽), cam2_path(왼쪽)를 기준으로 합니다.
            lines.append(line)
    return lines

def image_generator(samples, data_dir, batch_size=32):
    """
    학습 데이터를 배치(batch) 단위로 생성하는 제너레이터입니다.
    메모리 부족 문제 없이 대용량의 이미지 데이터를 학습할 수 있게 해줍니다.
    """
    num_samples = len(samples)
    while True: # 모델이 학습하는 동안 무한히 데이터를 생성합니다.
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # --- [중요] 오른쪽 카메라(cam1) 이미지만 사용 ---
                # 헤더를 right_cam_path로 바꿨다면 이 부분을 수정하세요.
                # 예: image_path = os.path.join(data_dir, batch_sample[0].strip())
                image_name = batch_sample[0].strip() 
                image_path = os.path.join(data_dir, image_name)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                
                # [중요] 조향 값(steering)만 사용
                angle = float(batch_sample[2]) 
                
                images.append(image)
                angles.append(angle)
                
                # --- 데이터 증강 (Data Augmentation) ---
                # 이미지를 좌우로 뒤집고, 조향 값의 부호를 반대로 하여 데이터를 2배로 늘립니다.
                images.append(cv2.flip(image, 1))
                angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

def create_nvidia_model():
    """
    NVIDIA의 "End-to-End Learning for Self-Driving Cars" 논문 기반 모델 생성
    """
    model = Sequential()
    # 1. Normalization Layer: 이미지 픽셀 값을 -1 ~ 1 범위로 정규화
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
    
    # 2. Cropping Layer: 이미지 상단의 불필요한 부분(하늘, 나무 등)과 하단의 차량 일부를 잘라냄
    # Keras에 기본 Cropping2D가 있지만, Lambda로도 구현 가능
    # 지금은 설명을 위해 생략하고, 전처리 단계에서 수행하는 것을 추천합니다.
    
    # 3. Convolutional Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # 4. Flatten Layer
    model.add(Flatten())
    
    # 5. Fully Connected Layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1)) # 최종 출력: 조향 값 하나
    
    return model

if __name__ == '__main__':
    # 데이터 경로 설정
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('fsds_agent')
    data_dir = os.path.join(pkg_path, "data")
    log_path = os.path.join(data_dir, 'driving_log_passive.csv')

    # 1. 데이터 로드 및 분할
    samples = load_driving_log(log_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print(f"Total data points: {len(samples)}")
    print(f"Training data points: {len(train_samples)}")
    print(f"Validation data points: {len(validation_samples)}")

    # 2. 제너레이터 생성
    batch_size = 32
    train_generator = image_generator(train_samples, data_dir, batch_size=batch_size)
    validation_generator = image_generator(validation_samples, data_dir, batch_size=batch_size)

    # 3. 모델 생성 및 컴파일
    model = create_nvidia_model()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    model.summary() # 모델 구조 출력

    # 4. 모델 학습
    epochs = 20
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_samples) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_samples) // batch_size,
        epochs=epochs
    )

    # 5. 학습된 모델 저장
    model.save('model.h5')
    print("Model saved as model.h5")