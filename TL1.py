import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.title("📸 Image Processing Demo")

# Upload ảnh
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ================== Group chọn ==================
group = st.radio("Choose method group", ["Basic Transformations", "Manual Methods"])

# ================== Menu theo nhóm ==================
if group == "Basic Transformations":
    method = st.selectbox(
        "Choose a transformation method",
        ["Negative Image", "Log Transformation", "Gamma Correction", "Piecewise-linear Transformation"]
    )
elif group == "Manual Methods":
    method = st.selectbox(
        "Choose a manual method",
        ["Histogram Equalization", "CLAHE (Custom)", "Compare Histograms (Equalization vs CLAHE)"]
    )

# ================== HÀM HISTOGRAM EQUALIZATION ==================
def Histogram_Equalization(img):
    h, w = img.shape
    N = h * w
    L = 256
    hist = [0] * L
    for i in range(h):
        for j in range(w):
            p_value = img[i][j]
            hist[p_value] += 1

    pdf = [h_ / N for h_ in hist]

    cdf = [0] * L
    cal = 0
    for i in range(L):
        cal += pdf[i]
        cdf[i] = cal

    Tx = [int((L - 1) * i + 0.5) for i in cdf]

    eq_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            eq_img[i, j] = Tx[img[i, j]]
    return eq_img

# ================== HÀM CLAHE THỦ CÔNG ==================
def custom_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    height, width = image.shape
    tile_height = height // tile_grid_size[0]
    tile_width = width // tile_grid_size[1]

    output = np.zeros_like(image, dtype=np.uint8)

    def process_tile(tile, clip_limit, num_bins=256):
        hist, _ = np.histogram(tile.ravel(), bins=num_bins, range=(0, 255))
        total_pixels = tile.size
        clip_limit = max(1, clip_limit * total_pixels / num_bins)

        excess = 0
        clipped_hist = np.zeros_like(hist)
        for i in range(num_bins):
            if hist[i] > clip_limit:
                excess += hist[i] - clip_limit
                clipped_hist[i] = int(clip_limit)
            else:
                clipped_hist[i] = hist[i]

        redistribution = int(excess // num_bins)
        clipped_hist += redistribution

        cdf = clipped_hist.cumsum()
        cdf = cdf / cdf[-1] * (num_bins - 1)
        mapping = np.round(cdf).astype(np.uint8)
        return mapping

    mappings = []
    for i in range(tile_grid_size[0]):
        row_mappings = []
        for j in range(tile_grid_size[1]):
            tile = image[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
            mapping = process_tile(tile, clip_limit)
            row_mappings.append(mapping)
        mappings.append(row_mappings)

    for y in range(height):
        for x in range(width):
            tile_i = min(y // tile_height, tile_grid_size[0] - 1)
            tile_j = min(x // tile_width, tile_grid_size[1] - 1)

            tile_y = (y % tile_height) / tile_height
            tile_x = (x % tile_width) / tile_width

            top_left = mappings[tile_i][tile_j]
            top_right = mappings[tile_i][min(tile_j + 1, tile_grid_size[1] - 1)]
            bottom_left = mappings[min(tile_i + 1, tile_grid_size[0] - 1)][tile_j]
            bottom_right = mappings[min(tile_i + 1, tile_grid_size[0] - 1)][min(tile_j + 1, tile_grid_size[1] - 1)]

            pixel_value = image[y, x]
            val1 = top_left[pixel_value] * (1 - tile_x) + top_right[pixel_value] * tile_x
            val2 = bottom_left[pixel_value] * (1 - tile_x) + bottom_right[pixel_value] * tile_x
            interpolated_value = val1 * (1 - tile_y) + val2 * tile_y

            output[y, x] = np.round(interpolated_value).astype(np.uint8)

    return output


# ================== XỬ LÝ ==================
if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("RGB"))

    if group == "Manual Methods":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if method == "Histogram Equalization":
            result = Histogram_Equalization(gray)
            st.image([gray, result], caption=["Original Gray", "Equalization"], use_container_width=True)

        elif method == "CLAHE (Custom)":
            clip = st.slider("Clip Limit", 1.0, 10.0, 4.0, 0.5)
            grid = st.slider("Tile Grid Size", 2, 16, 8, 1)
            result = custom_clahe(gray, clip_limit=clip, tile_grid_size=(grid, grid))
            st.image([gray, result], caption=["Original Gray", "CLAHE"], use_container_width=True)

        elif method == "Compare Histograms (Equalization vs CLAHE)":
        # Thêm thanh trượt để điều chỉnh CLAHE
            clip = st.slider("Clip Limit (CLAHE)", 1.0, 10.0, 4.0, 0.5)
            grid = st.slider("Tile Grid Size (CLAHE)", 2, 16, 8, 1)

            # Histogram Equalization
            img_h = Histogram_Equalization(gray)

            # CLAHE với tham số tùy chỉnh
            img_clahe = custom_clahe(gray, clip_limit=clip, tile_grid_size=(grid, grid))

            # Vẽ histogram so sánh
            fig = plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.hist(gray.ravel(), bins=256, range=(0, 255), color='gray')
            plt.title('Histogram ảnh gốc')
            plt.xlabel('Cường độ pixel')
            plt.ylabel('Số lượng')

            plt.subplot(1, 3, 2)
            plt.hist(img_h.ravel(), bins=256, range=(0, 255), color='blue')
            plt.title('Histogram Equalization')
            plt.xlabel('Cường độ pixel')
            plt.ylabel('Số lượng')

            plt.subplot(1, 3, 3)
            plt.hist(img_clahe.ravel(), bins=256, range=(0, 255), color='green')
            plt.title(f'Histogram CLAHE (clip={clip}, grid={grid}x{grid})')
            plt.xlabel('Cường độ pixel')
            plt.ylabel('Số lượng')

            plt.tight_layout()
            st.pyplot(fig)

            # Hiển thị ảnh
            st.image([gray, img_h, img_clahe],
                caption=["Original Gray", "Equalization", f"CLAHE (clip={clip}, grid={grid}x{grid})"],
                use_container_width=True)


    elif group == "Basic Transformations":
        if method == "Negative Image":
            result = 255 - img
            st.image([img, result], caption=["Original", "Negative"], use_container_width=True)

        elif method == "Log Transformation":
            img_float = img.astype(np.float32)
            c = 255 / (np.log(1 + np.max(img_float)))
            log_transformed = c * np.log(1 + img_float)
            result = np.array(log_transformed, dtype=np.uint8)
            st.image([img, result], caption=["Original", "Log Transformation"], use_container_width=True)

        elif method == "Gamma Correction":
            gamma = st.slider("Gamma Value", 0.1, 5.0, 1.0, 0.1)
            gamma_corrected = np.array(255 * ((img / 255) ** gamma), dtype=np.uint8)
            st.image([img, gamma_corrected], caption=["Original", f"Gamma = {gamma}"], use_container_width=True)

        elif method == "Piecewise-linear Transformation":
            # ví dụ đơn giản: contrast stretching
            r1, s1 = 50, 0
            r2, s2 = 200, 255
            def piecewise_linear(val):
                if val < r1:
                    return (s1 / r1) * val
                elif val < r2:
                    return ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
                else:
                    return ((255 - s2) / (255 - r2)) * (val - r2) + s2
            vectorized = np.vectorize(piecewise_linear)
            result = vectorized(img).astype(np.uint8)
            st.image([img, result], caption=["Original", "Piecewise-linear"], use_container_width=True)

# streamlit run TL1.py