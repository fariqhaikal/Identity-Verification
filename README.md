# Identity Verification System

This project performs identity verification using OCR and facial recognition.

## Features

- Extracts ID information using Tesseract OCR
- Cleans and matches data with a reference CSV
- Compares face from ID and selfie using `face_recognition`
- Gives a final verification result with similarity scores

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
