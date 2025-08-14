from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from gtts import gTTS
import tempfile
import os

app = Flask(__name__)
CORS(app)

MODEL_NAME = "ibm-granite/granite-3.1-3b-a800m-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("Loading model with optimized settings (device_map=auto, float16, offload)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",            # Auto-place on GPU/CPU as needed
    torch_dtype=torch.float16,    # Use FP16 for less VRAM
    low_cpu_mem_usage=True,       # Reduce CPU RAM during load
    offload_folder="offload"      # Folder to store offloaded layers
)
model.eval()
print("Model loaded successfully.")


def generate_text(prompt, max_tokens=300, temperature=0.7):
    """
    Generate text from the Granite model using a chat prompt style.
    Keep inputs on CPU so device_map can handle placement automatically.
    """
    chat = [{"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt")  # keep inputs on CPU

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated


@app.route('/generate', methods=['POST'])
def generate():
    """
    POST JSON: { "topic": "Your topic here", "test": true/false }
    If test: true, skip model and directly test TTS with a fixed string.
    """
    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' in request JSON"}), 400

    topic = data["topic"].strip()
    if not topic:
        return jsonify({"error": "Empty topic provided"}), 400

    test_mode = data.get("test", False)

    try:
        if test_mode:
            print("Test mode enabled — skipping model generation.")
            generated_text = "This is a test of text to speech conversion."
        else:
            print(f"Generating text for topic: {topic}")
            generated_text = generate_text(topic)
            print("Text generation complete.")

        if not generated_text.strip():
            return jsonify({"error": "Generated text is empty"}), 500

        # Convert to speech with gTTS
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "output.mp3")
        tts = gTTS(generated_text, lang='en')
        tts.save(audio_path)
        print(f"Audio saved to {audio_path}")

        return send_file(audio_path, as_attachment=True, download_name="audiobook.mp3")

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": "Failed to generate or convert audio",
            "details": str(e)
        }), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "EchoVerse IBM Granite 3.1 AI Audiobook Generator API is running."})


if __name__ == '__main__':
    # Make sure offload folder exists
    if not os.path.exists("offload"):
        os.makedirs("offload")
    app.run(debug=True)

