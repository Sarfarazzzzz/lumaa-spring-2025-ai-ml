# Movie Recommendation System

**Overview**

This project is a **content-based movie recommendation system** that suggests movies based on a user's textual input. It uses **TF-IDF vectorization** and **cosine similarity** to compare user preferences with movie metadata. The system combines multiple movie attributes to provide accurate recommendations.

---

## **Dataset**

- **Filename:** `tmdb-movies500.csv` (path hardcoded in the script)
- **Columns Used:**
  - `overview`: Movie plot summary.
  - `cast`: Main cast members.
  - `genres`: Genres associated with the movie.
  - `keywords`: Relevant keywords.
  - `tagline`: Movie tagline.
  - `original_title`: Used for displaying the recommendation results.
  - `release_date`: Displayed alongside recommendations.

---

##  **Setup Instructions**

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/Sarfarazzzzz/lumaa-spring-2025-ai-ml.git
cd <repository_folder>
```

### 2️⃣ **Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\\Scripts\\activate  # Windows
```

### 3️⃣ **Install Dependencies**

```bash
pip3 install -r requirements.txt
```

---

##  **Running the Recommendation System**

Run the script directly:

```bash
python3 CBR.py
```

### **Example Interaction:**

```
What do you feel like watching today ?? I love thrilling action movies set in space, with a comedic twist

Top Recommendations:
400 Days (10/29/15) - 20.20% Match
Equals (9/4/15) - 18.26% Match
The Lovers (2/13/15) - 15.36% Match
Mad Max: Fury Road (5/13/15) - 15.09% Match
Harbinger Down (8/7/15) - 14.61% Match
```

---

##  **Code Structure**

| Function Name           | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `load_dataset()`        | Loads the dataset with Latin1 encoding.                                      |
| `combine_features()`    | Combines overview, cast, genres, keywords, tagline into a single text field. |
| `preprocess_text()`     | Cleans the text by lowercasing and removing punctuation.                     |
| `preprocess_dataset()`  | Applies preprocessing to combined features.                                  |
| `vectorize_text()`      | Transforms combined features using TF-IDF.                                   |
| `recommend_items()`     | Computes cosine similarity and returns top matches.                          |
| `main_recommendation()` | Handles the full recommendation pipeline.                                    |
| `test()`                | Wrapper to run recommendations based on user input.                          |

---

## **Key Features & Functionality**

- Uses **TF-IDF Vectorization** with bigram support (`ngram_range=(1, 2)`) for better context.
- Cleans text thoroughly to improve similarity matching.
- Displays **similarity scores** as a percentage for clarity.
- Accepts **user input** directly via the command line for real-time recommendations.

---


##  **Video Demonstration**

***https://drive.google.com/file/d/1yvbUH2F2kAl3jbFcfhgtm2DLDuKiaxsT/view?usp=drive_link***

---

**Salary expectation per month:** 

$3500 per month ($20-$25 per hour)

**Contact**
MailId : m.shaik@gwu.edu

