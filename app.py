from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from gpt import GPTModel, GPT_CONFIG_774M
from until import format_input, text_to_token_ids, token_ids_to_text, tokenizer, generate

app = Flask(__name__)
CORS(app)

# Initialize device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt = GPTModel(GPT_CONFIG_774M)
gpt.load_state_dict(torch.load("gita774M(2).pt", map_location=device))
gpt.to(device)
gpt.eval()

# Home route: serve chatbot UI
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route: handle AJAX POST request
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")

        # Format input in instruction format
        prompt = format_input({"instruction": user_input, "input": ""})

        # Tokenize input
        token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Generate response using model
        generated = generate(
            model=gpt,
            idx=input_tensor,
            max_new_tokens=100,
            context_size=1024,
            temperature=0.9,
            top_k=50,
            eos_id=tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        )

        # Decode output
        output_text = token_ids_to_text(generated, tokenizer)

        # Remove original prompt from output
        final_output = output_text[len(prompt):].strip()

        # Remove "### Response:" prefix if present
        if final_output.lower().startswith("### response:"):
            final_output = final_output[len("### response:"):].strip()

        return jsonify({"response": final_output})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"response": "An internal error occurred."}), 500

# Start the server
if __name__ == "__main__":
    app.run(port=5000)
