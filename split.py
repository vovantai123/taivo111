from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import io
import zipfile
import re
import os

# ⚙️ Cấu hình Tesseract cho Windows
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = Flask(__name__)

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Kiểm tra ảnh hợp lệ
        if img is None:
            return jsonify({"error": "Không thể đọc ảnh"}), 400

        # Chuyển grayscale và nhị phân
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Tìm contour các khung text
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        # --- OCR từng khung phát hiện ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                roi = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 9
                )

                text = pytesseract.image_to_string(
                    roi, lang="eng+fra+spa", config="--oem 3 --psm 4"
                )

                # Chuẩn hóa lỗi OCR phổ biến
                replacements = {"&": "À", "¢": "ç", "|": "l", "¢¢": "é"}
                for wrong, right in replacements.items():
                    text = text.replace(wrong, right)

                results.append((y, x, w, h, text.strip()))

        # --- Sắp xếp block theo thứ tự trên - dưới, trái - phải ---
        results.sort(key=lambda r: (r[0], r[1]))

        # --- Tạo file ZIP kết quả ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            block_index = 0

            for (y, x, w, h, text) in results:
                # Giới hạn vùng cắt, mở rộng một chút
                y_top = max(y - 20, 0)
                y_bottom = min(y + h + 150, img.shape[0])
                x_min = max(x - 20, 0)
                x_max = min(x + w + 20, img.shape[1])

                # Cắt block riêng
                crop = img[y_top:y_bottom, x_min:x_max]

                # OCR trong block để tìm CARE CODE
                roi_code = gray[y_top:y_bottom, x_min:x_max]
                code_text = pytesseract.image_to_string(
                    roi_code, lang="eng", config="--psm 6"
                ).strip()

                # Tìm mã CARE CODE nếu có
                match = re.search(r"(CARE\s*\d+)", code_text.upper())
                if match:
                    filename = match.group(1).replace(" ", "_") + ".jpg"
                else:
                    filename = f"block_{block_index + 1}.jpg"

                # Ghi ảnh vào zip
                _, enc = cv2.imencode(".jpg", crop)
                zipf.writestr(filename, enc.tobytes())

                print(f"[INFO] Saved block {block_index + 1}: {filename}")
                block_index += 1

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name="care_blocks.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
