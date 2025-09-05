import google.generativeai as genai
from PIL import Image

genai.configure(api_key="your-api-key")
image = Image.open("img3.jpg")

prompt = "Identify the plant disease in this image and suggest possible cures and pesticides."

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content([prompt, image])
print("\n=== Gemini Response ===\n")
print(response.text)
