import os
import re
import cv2
import pytesseract
import pandas as pd
import face_recognition
import numpy as np
from scipy.spatial.distance import cosine
import Levenshtein as lev

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# ------------------ OCR Functions ------------------

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    rotated = cv2.rotate(adjusted, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated

def define_regions(image):
    h, w = image.shape[:2]
    return {
        "Title": image[0:int(h * 0.2), 0:int(w * 0.6)],
        "ID Number": image[int(h * 0.22):int(h * 0.3), 0:int(w * 0.5)],
        "Name": image[int(h * 0.64):int(h * 0.71), 0:int(w * 0.5)],
        "Address": image[int(h * 0.7):, 0:int(w * 0.6)],
    }

def extract_text_from_regions(regions):
    data = {}
    for field, region in regions.items():
        text = pytesseract.image_to_string(region, lang='eng')
        data[field] = text.strip().replace("\n", " ") if field == "Address" else text.strip()
    return data

def clean_ocr_data(data):
    corrections = {
        "KAN": "KAD", "PENGENAI": "PENGENALAN", "MEAN": "MALAYSIA",
        "LAY": "MALAYSIA", "NTIFV": "IDENTITY", "Cari": "CARD",
    }
    for key, value in data.items():
        if isinstance(value, str):
            for wrong, correct in corrections.items():
                value = value.replace(wrong, correct)
            data[key] = value
            if key == "Address":
                data[key] = value.replace("\n", " ")
        else:
            data[key] = str(value)
    
    match = re.search(r"\d{6}-\d{2}-\d{4}", data.get("ID Number", ""))
    data["ID Number"] = match.group() if match else "Not Found"
    return data

def calculate_similarity(extracted, reference):
    similarity = 0
    exact_match = True
    for key, ref_val in reference.items():
        ext_val = extracted.get(key, "")
        if isinstance(ref_val, str) and isinstance(ext_val, str):
            lev_dist = lev.distance(ext_val, ref_val)
            max_len = max(len(ext_val), len(ref_val), 1)
            field_sim = (1 - lev_dist / max_len) * 100
            similarity += field_sim
            if lev_dist > 0:
                exact_match = False
    average_sim = similarity / len(reference) if reference else 0
    return exact_match, average_sim

def match_reference(csv_path, extracted_data):
    df = pd.read_csv(csv_path)
    best_match = None
    best_score = 0
    for _, row in df.iterrows():
        ref = {
            "Title": row["Malaysia_id_card"],
            "ID Number": row["id"],
            "Name": row["Name"],
            "Address": row["Address"]
        }
        _, score = calculate_similarity(extracted_data, ref)
        if score > best_score:
            best_score = score
            best_match = ref
    return best_match, best_score

# ------------------ Face Recognition ------------------

def get_face_encoding(image):
    locations = face_recognition.face_locations(image)
    if locations:
        return face_recognition.face_encodings(image, known_face_locations=locations)[0]
    return None

def compare_faces(enc1, enc2):
    return 1 - cosine(enc1, enc2)

def process_face_verification(id_photo_path, selfie_path):
    id_img = face_recognition.load_image_file(id_photo_path)
    selfie_img = face_recognition.load_image_file(selfie_path)
    id_bgr = cv2.cvtColor(id_img, cv2.COLOR_RGB2BGR)
    id_rotated = cv2.rotate(id_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    id_rgb = cv2.cvtColor(id_rotated, cv2.COLOR_BGR2RGB)

    id_encoding = get_face_encoding(id_rgb)
    selfie_encoding = get_face_encoding(selfie_img)

    if id_encoding is None or selfie_encoding is None:
        return None  # Face not detected
    return compare_faces(id_encoding, selfie_encoding)

# ------------------ Main Unified Flow ------------------

def main():
    current_dir = os.getcwd()
    id_image_path = os.path.join(current_dir, "Datasets/id_4.jpeg")
    id_face_path = os.path.join(current_dir, "Datasets/id.jpeg")
    selfie_path = os.path.join(current_dir, "Datasets/picture.jpeg")
    csv_path = os.path.join(current_dir, "Datasets/database.csv")

    print("Starting OCR and Face Verification...\n")

    # OCR
    image = preprocess_image(id_image_path)
    regions = define_regions(image)
    raw_data = extract_text_from_regions(regions)
    structured_data = clean_ocr_data(raw_data)

    print("📄 Extracted & Cleaned OCR Data:")
    for k, v in structured_data.items():
        print(f"{k}: {v}")

    reference, ocr_score = match_reference(csv_path, structured_data)
    print(f"\n📊 Best OCR Match Score: {ocr_score:.1f}%")

    # Face Verification
    similarity = process_face_verification(id_face_path, selfie_path)
    if similarity is not None:
        print(f"\n🧑‍🤝‍🧑 Face Similarity Score: {similarity * 100:.2f}%")
    else:
        print("\n❌ Face not detected in one of the images.")

    # Final Decision
    print("\n🔍 Final Identity Verification Result:")
    ocr_pass = ocr_score >= 85
    face_pass = similarity is not None and similarity >= 0.6

    if ocr_pass and face_pass:
        print("✅ Identity Verified")
    else:
        print("❌ Identity Verification Failed")
        if not ocr_pass:
            print("- OCR data mismatch or low similarity")
        if not face_pass:
            print("- Face mismatch or undetected")

if __name__ == "__main__":
    main()
