from PIL import Image
import numpy as np
import elice_utils

def main():
    """
    불러온 이미지를 RGB 값을 갖는 3차원 벡터로 변환합니다.
    """
    
    # 이미지를 불러옵니다.
    filenames = ["./data/aurora.jpg", "./data/raccoon.jpg"]
    img = Image.open(filenames[0])
    img = img.convert("RGB")
    
    # 이미지를 NumPy 배열로 변환합니다. (높이, 넓이, 3) 차원의 행렬로 변환하여 돌려줍니다.
    image_vector = np.asarray(img)
    prep_image_vector = image_vector
    
    #preprocess(image_vector)
    
    # K-means의 K값을 설정합니다.
    K = 32
    
    new_image, clusters, centroids = kmeans(prep_image_vector, K)
    new_image = postprocess(new_image)
    
    # 변환된 벡터의 타입을 처리된 이미지와 같이 8bits로 설정합니다.
    new_image = new_image.astype("uint8")
    # 데이터를 이미지로 다시 변환합니다. 
    new_img = Image.fromarray(new_image, "RGB")
    # 이미지를 저장하고 실행결과를 출력합니다.
    new_img.save("image1out.jpg")
    elice = elice_utils.EliceUtils()
    elice.send_image("image1out.jpg")
    
    # 점수를 확인합니다.
    print("Score = %.2f" % elice.calc_score(image_vector, new_image))
    return


def kmeans(image_vector, K = 32):
    """
    클러스터링을 시작합니다. pick_seed를 이용해 초기 Centroid를 구합니다.
    regress는 이용해 클러스터를 할당(allocate_cluster)하고 새로운 Centroid를 계산(recalc_centroids)합니다. 
    """
    
    clusters = np.zeros((image_vector.shape[:2]))
    centroids = np.array(pick_seed(image_vector, K))

    while True:
        new_clusters, new_centroids = regress(image_vector, centroids)
        if np.all(clusters == new_clusters):
            break
        clusters, centroids = new_clusters, new_centroids

    new_image = np.zeros(image_vector.shape)

    for i in range(K):
        new_image[clusters == i] = centroids[i]

    return new_image, clusters, centroids


def regress(image_vector, centroids):
    new_clusters = allocate_cluster(image_vector, centroids)
    new_centroids = recalc_centroids(image_vector, new_clusters, centroids.shape[0])
    
    return new_clusters, new_centroids


def pick_seed(image_vector, K):
    """
    image_vector로부터 K개의 점을 선택하여 리턴해주세요!    
    """
    centroids = np.zeros((K, image_vector.shape[2]))
    
    for x in range(len(centroids)):
        a=np.random.choice(255)
        b=np.random.choice(174)
        c=np.random.choice(110)
        centroids[x] = [a,b,c]
        
    return centroids

def allocate_cluster(image_vector, centroids):
    height, width, _ = image_vector.shape
    print(image_vector.shape)
    clusters = np.zeros((height, width))
    
    """
    주어진 Centroid에 따라 새로운 클러스터에 할당해주세요.
    
    예를들어, 0행 0열의 pixel이 3번 Centroid에 가깝다면, clusters[0][0] = 3 이라고 채웁니다.
    
    """
    
    return clusters
    
def recalc_centroids(image_vector, clusters, K):
    centroids = np.zeros((K, image_vector.shape[2]))
    """
    Cluster별로 새로운 centroid를 구해서 되돌려줍니다.
    
    """
    
    return centroids

def postprocess(image_vector):
    """
    이미지 후처리 함수
    
    이 함수를 필히 작성하지 않아도 동작합니다.
    """
    return image_vector
def preprocess(x):

    return x

if __name__ == "__main__":
    main()