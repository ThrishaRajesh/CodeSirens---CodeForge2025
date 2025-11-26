import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from io import BytesIO
import base64

from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')


# --------------------
# Helper Functions
# --------------------
def resize_images(imgA, imgB, max_dim=800):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    scale = min(max_dim / wA, max_dim / hA, max_dim / wB, max_dim / hB, 1)
    imgA = cv2.resize(imgA, (int(wA * scale), int(hA * scale)))
    imgB = cv2.resize(imgB, (int(wB * scale), int(hB * scale)))
    h = min(imgA.shape[0], imgB.shape[0])
    w = min(imgA.shape[1], imgB.shape[1])
    imgA = imgA[:h, :w]
    imgB = imgB[:h, :w]
    return imgA, imgB


def compare_absdiff(grayA, grayB, threshold):
    diff = cv2.absdiff(grayA, grayB)
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return th


def compare_ssim_mask(grayA, grayB):
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    inv = cv2.bitwise_not(diff)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th


def postprocess(binary, min_area=1000):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
    return mask


def damage_score_color(pct):
    if pct < 5:
        return 0, (0, 255, 0)
    elif pct < 20:
        return 1, (0, 255, 255)
    elif pct < 50:
        return 2, (0, 165, 255)
    else:
        return 3, (0, 0, 255)


def overlay_damage_colored(after, mask, color, alpha=0.45):
    overlay = after.copy().astype("float32")
    color_layer = np.zeros_like(after, dtype=np.uint8)
    color_layer[:] = color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * after[mask_bool].astype("float32") + alpha * color_layer[mask_bool]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 0), 2)
    return overlay.astype("uint8")


def cv2_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = False
    method = "AbsDiff"
    threshold = 30

    pct = 0.0
    score = 0
    level = ""
    before_img_b64 = ""
    overlay_img_b64 = ""

    if request.method == "POST":
        before_file = request.files.get("before")
        after_file = request.files.get("after")
        method = request.form.get("method", "AbsDiff")

        if method == "AbsDiff":
            try:
                threshold = int(request.form.get("threshold", 30))
            except ValueError:
                threshold = 30
        else:
            threshold = None

        if not before_file or not after_file:
            error = "Please upload both BEFORE and AFTER images."
        else:
            try:
                # Load via PIL and convert to numpy
                imgA = Image.open(before_file.stream).convert("RGB")
                imgB = Image.open(after_file.stream).convert("RGB")
                imgA = np.array(imgA)
                imgB = np.array(imgB)

                # Convert to BGR for OpenCV
                imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2BGR)
                imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2BGR)

                # Resize + gray + blur
                imgA, imgB = resize_images(imgA, imgB)
                grayA = cv2.GaussianBlur(cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                grayB = cv2.GaussianBlur(cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY), (5, 5), 0)

                # Comparison
                if method == "AbsDiff":
                    raw_mask = compare_absdiff(grayA, grayB, threshold)
                else:
                    raw_mask = compare_ssim_mask(grayA, grayB)

                mask = postprocess(raw_mask, min_area=1000)

                total = mask.size
                damaged = cv2.countNonZero(mask)
                pct = (damaged / total) * 100 if total > 0 else 0.0

                score, color = damage_score_color(pct)
                overlay = overlay_damage_colored(imgB, mask, color)

                labels = ["No damage", "Minor", "Major", "Destroyed"]
                level = labels[score]

                before_img_b64 = cv2_to_base64(imgA)
                overlay_img_b64 = cv2_to_base64(overlay)
                result = True

            except Exception as e:
                error = f"Error processing images: {str(e)}"

    return render_template(
        "index.html",
        error=error,
        result=result,
        pct=pct,
        score=score,
        level=level,
        before_img=before_img_b64,
        overlay_img=overlay_img_b64,
        method=method,
        threshold=threshold,
    )


if __name__ == "__main__":
    app.run(debug=True)
