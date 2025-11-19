# Face Recognition Event Matcher 📸

This application analyzes a user's selfie (query image) and retrieves all similar images from a large volume of event photos. It simplifies the process of finding specific people in massive photo albums, perfect for weddings, parties, or corporate events.

## 🚀 Features

* **Smart Face Matching:** Uses InsightFace to detect and match faces with high accuracy.
* **Event Management:** Create unique events and upload bulk folders of raw images.
* **Fast Processing:** Parallel processing ensures quick uploads and fast search results.
* **Privacy Focused:** Deletes raw data securely when an event is removed.
* **Download All:** Users can download all their matched photos in a single ZIP file.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, InsightFace (Computer Vision)
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
* **Storage:** Supabase (Object Storage)

---
Also Attached a notebook where it was trained on cnn and hog model explore them too

## ⚙️ Installation & Setup Guide

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/your-username/Face_recognition_model.git
cd Face_recognition_model
```

### 2. Set Up a Virtual Environment (Recommended)

It is best practice to run Python projects in a virtual environment to avoid conflicts.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (.env)

This project requires a **Supabase** account for storing images.

1. Open the `.env` file located in the root directory.
2. You will see placeholders for the API keys.
3. Replace the text `replace_your_url_here` and `replace_your_key_here` with your actual Supabase credentials.

**Example `.env` file:**

```ini
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key
SUPABASE_BUCKET=event-photos
```

> **Note:** You can find these keys in your Supabase Dashboard under **Project Settings > API**.

---

## 🏃‍♂️ How to Run

### 1. Start the Backend Server

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

You should see a message saying the server is running at `http://127.0.0.1:8000`.

### 2. Launch the Frontend

Simply open the `index.html` file in your web browser (Chrome, Edge, Firefox, etc.).

1. Go to the project folder.
2. Double-click `index.html`.
3. You can now upload folders and search for faces!

---

## 📂 Project Structure

```
Face_recognition_model/
│
├── main.py              # The FastAPI Backend logic
├── index.html           # The Frontend Interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API Keys)
└── README.md            # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
