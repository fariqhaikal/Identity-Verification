# 🆔 Identity Verification System (Simulation)

This project simulates an **identity verification system** using a combination of **OCR (Optical Character Recognition)** and **facial recognition** to validate identity documents.

---

## 🛂 What This Project Does

This program demonstrates a **simulated identity verification** process by:

1. **Extracting text** (e.g., ID number, name, address) from an image of an ID card using Tesseract OCR.
2. **Matching a selfie** against the face on the ID photo using facial recognition.
3. **Comparing the extracted data** against a predefined reference database to simulate identity validation.

The idea is to replicate a typical identity verification step you'd find in online registration or KYC (Know Your Customer) processes.

---

## 📁 Project Structure


---

## 🧾 Database Structure

The simulation matches extracted ID data to records in a pre-built database file named `database.csv`.

This file must include the following columns:

| Column Name         | Description                                     |
|---------------------|-------------------------------------------------|
| `malaysian_id_card` | Title from the ID (e.g., "MyKad")               |
| `id`                | National ID number (e.g., `123456-78-9012`)     |
| `name`              | Full name as shown on the ID                    |
| `address`           | Residential address from the ID card            |

The program compares OCR output with each row in the database to evaluate how closely the ID matches stored records.

---

## 🚀 How to Run

### Step 1: Install dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
