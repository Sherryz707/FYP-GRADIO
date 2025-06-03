from fastai.vision.all import load_learner
from PIL import Image
import gradio as gr

# Load both models
alphabet_model = load_learner('alphabets.pkl', cpu=True)
digit_model = load_learner('digits.pkl', cpu=True)

# Predict functions
def predict_alphabet(image):
    image = image.resize((224, 224))
    pred, pred_idx, probs = alphabet_model.predict(image)
    return {str(alphabet_model.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

def predict_digit(image):
    image = image.resize((224, 224))
    pred, pred_idx, probs = digit_model.predict(image)
    return {str(digit_model.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

# Define two interfaces
alphabet_interface = gr.Interface(
    fn=predict_alphabet,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Alphabet Sign Recognition",
    description="Upload an image to predict its alphabet sign."
)

digit_interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Digit Sign Recognition",
    description="Upload an image to predict its digit sign."
)

# Combine in tabs
demo = gr.TabbedInterface(
    interface_list=[alphabet_interface, digit_interface],
    tab_names=["Alphabets", "Digits"]
)

# Run locally
if __name__ == "__main__":
    demo.launch()
