import google.generativeai as genai
from openai import OpenAI
from groq import Groq
import openai
import groq

def generate_answer(context, question, provider="Gemini", model_name="gemini-2.5-flash", api_key=None):
    if not api_key:
        raise ValueError(f"API key is missing for provider '{provider}'. Please check your configuration.")

    prompt = f"""You are a helpful assistant.

If the user's message is a greeting, farewell, or polite chitchat (e.g., "hello", "hi", "how are you", "thank you", "hey"), respond politely, naturally, and invite them to ask questions about the uploaded document. Do not check or require the context for greetings.

Otherwise, answer the user's question ONLY using the provided context. If the context does not contain the answer or is empty, clearly state: "I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}
"""

    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if not response.text:
                raise ValueError("Received empty response from Gemini API.")
            return response.text

        elif provider == "ChatGPT (OpenAI)":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content

        elif provider == "Groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except openai.AuthenticationError:
        raise RuntimeError("OpenAI Authentication Failed: Please verify that your OpenAI API key is correct and active.")
    except openai.RateLimitError:
        raise RuntimeError("OpenAI Rate Limit Exceeded: You have hit the request quota limit. Please wait a bit before trying again.")
    except openai.APIError as e:
        raise RuntimeError(f"OpenAI API Error: {str(e)}")

    except groq.AuthenticationError:
        raise RuntimeError("Groq Authentication Failed: Please verify that your Groq API key is correct and active.")
    except groq.RateLimitError:
        raise RuntimeError("Groq Rate Limit Exceeded: You have hit the request quota limit. Please wait a bit before trying again.")
    except groq.APIError as e:
        raise RuntimeError(f"Groq API Error: {str(e)}")

    except Exception as e:
        # Catch Gemini errors or general exceptions safely without exposing stack traces
        err_msg = str(e)
        if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
            raise RuntimeError("Gemini Authentication Failed: Please verify that your Gemini API key is correct and active.")
        elif "quota" in err_msg.lower() or "limit" in err_msg.lower():
            raise RuntimeError("Gemini Rate Limit Exceeded: You have hit the request quota limit.")
        raise RuntimeError(f"LLM Provider Error ({provider}): {err_msg}")