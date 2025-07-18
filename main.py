import os
import json
import tempfile
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import easyocr

# Load environment variables
load_dotenv()

# Set system prompt
SYSTEM_PROMPT = """
You will receive raw nutrition-label text (e.g. "Total Sugars: 15.1 g; Saturated Fat: 0.0 g; Sodium: 3.3 mg; Fiber: —; Protein: 0.0 g; Energy: 60.8 kcal").

Extract and return the following fields in **strict JSON format** (no explanations, no extra text):

- sugar    : float (grams of sugar)
- sat_fat  : float (grams of saturated fat)
- sodium   : int   (milligrams of sodium)
- fiber    : float (grams of dietary fiber; use 0.0 if missing or marked as —)
- protein  : float (grams of protein)
- calories : float (kcal)

Return the result exactly like this:

{
  "sugar": 12.0,
  "sat_fat": 4.5,
  "sodium": 250,
  "fiber": 3.0,
  "protein": 7.0,
  "calories": 180.0
}
"""

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Create FastAPI app
app = FastAPI(title="Nutrition Scoring API")

# Enable CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Nutrition schema
class NutritionInfo(BaseModel):
    sugar: float
    sat_fat: float
    sodium: int
    fiber: float
    protein: float
    calories: float

# Extract text from image using EasyOCR
def extract_text_from_image(image_path: str) -> str:
    try:
        results = reader.readtext(image_path)
        return " ".join(text for (bbox, text, prob) in results if prob > 0.5).strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from image: {str(e)}")

# Clean and parse LLM response JSON
def parse_nutrition_json(json_str: str) -> dict:
    clean = json_str.strip("```json").strip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

# Calculate score from nutrition values
def score_food(label_dict):
    max_vals = {
        "sugar": 30.0,
        "sat_fat": 15.0,
        "sodium": 1000.0,
        "calories": 500.0,
        "fiber": 10.0,
        "protein": 20.0
    }

    weights = {
        "sugar": -0.25,
        "sat_fat": -0.20,
        "sodium": -0.20,
        "calories": -0.15,
        "fiber": +0.10,
        "protein": +0.10
    }

    norm = {
        key: min(label_dict.get(key, 0), max_vals[key]) / max_vals[key]
        for key in max_vals
    }

    score_raw = sum(norm[k] * weights[k] for k in weights)
    return max(0, min(100, (score_raw + 1) * 50))

# Classify product by score
def get_class(score: float) -> str:
    if score < 20:
        return "Very Harmful"
    elif score < 40:
        return "Harmful"
    return "Safe"

# Call OpenRouter LLM
def call_llm(system_prompt: str, user_content: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content

# Main API endpoint
@app.post("/analyze", summary="Analyze nutrition from uploaded image and return classification")
async def analyze_nutrition(file: UploadFile = File(...)):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        extracted_text = extract_text_from_image(temp_path)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the image")

        formatted_json = call_llm(SYSTEM_PROMPT, extracted_text)
        nutrition_data = parse_nutrition_json(formatted_json)

        score = score_food(nutrition_data)
        classification = get_class(score)

        if classification == "Safe":
            message = "Safe product"
            better_product = "This is a better product"
        else:
            advice_prompt = f"""
            You are a nutrition coach. Given this nutrition data: {json.dumps(nutrition_data)}
            and classification: {classification}

            Respond with a JSON object containing:
            - "message": A brief 1-2 sentence explanation of why this product is {classification.lower()}
            - "better_product": Name of a specific healthier alternative product

            Only return the JSON, no extra text.
            """

            llm_response = call_llm(advice_prompt, "")
            try:
                advice_data = json.loads(llm_response.strip("```json").strip("```").strip())
                message = advice_data.get("message", f"This product is {classification.lower()} due to nutritional content")
                better_product = advice_data.get("better_product", "Consider a healthier alternative")
            except:
                message = f"This product is {classification.lower()} due to high levels of harmful nutrients"
                better_product = "Consider a product with lower sugar, sodium, and saturated fat"

        return {
            "class": classification,
            "message": message,
            "better_product": better_product,
            "extracted_text": extracted_text,
            "nutrition_data": nutrition_data,
            "score": round(score, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# Optional (for raw text-based input in future)
class NutritionRequest(BaseModel):
    nutrition_text: str
