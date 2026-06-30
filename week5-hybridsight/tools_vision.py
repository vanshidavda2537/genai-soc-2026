import os
import base64
from groq import Groq
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

client=Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def image_to_data_uri(filepath: str) -> str:
    """
    Convert an image file into a Base64 Data URI.
    """

    with open(filepath, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"

@tool
def describe_image(image_path: str) -> str:
    """
    Describe the content of an uploaded image.

    Use this tool whenever the user uploads an image
    and asks questions about it.
    """

    try:

        image_data = image_to_data_uri(image_path)

        response = client.chat.completions.create(

            model="llama-3.2-11b-vision-preview",

            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ]

        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Could not process image: {e}"