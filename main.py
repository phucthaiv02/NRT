# app.py
from handler.asset import load_models
from handler.bbox import get_patch

import gradio as gr
from transformers import pipeline
import torch
import traceback
from PIL import Image, ImageDraw

device = "cuda" if torch.cuda.is_available() else "cpu"

translator_1 = None
translator_2 = None
detector = None
recognizer = None
boxes_result = None
text_result = None


def load_all_model():
    global translator_1, translator_2, detector, recognizer
    translator_1 = pipeline(
        "translation", model="phucthaiv02/finetuned_nllb", device=device)
    translator_2 = pipeline(
        "translation", model="phucthaiv02/finetuned_nllb_2", device=device)

    detector, recognizer = load_models()


def run_translation_model(text_to_translate, mode):
    if not text_to_translate:
        return ""

    if mode == "Phiên âm":
        translator = translator_1
    else:
        translator = translator_2

    translation_result = translator(
        text_to_translate.split("\n"), src_lang="zho_Hant", tgt_lang="vie_Latn", clean_up_tokenization_spaces=True)

    translated_text = '\n'.join(
        [translation_result[i]['translation_text'] for i in range(len(translation_result))])

    splitted = translated_text.split('  ')
    ax = ' '.join([''.join(text.split(' ')) for text in splitted])
    return ax


def draw_boxes_on_image(raw_image):
    global boxes
    boxes = detector.predict_one_page(raw_image)
    image = Image.fromarray(raw_image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min = min(point[0] for point in box)
        y_min = min(point[1] for point in box)
        x_max = max(point[0] for point in box)
        y_max = max(point[1] for point in box)
        draw.rectangle([x_min, y_min, x_max, y_max],
                       outline="red", width=2)
    return image


def draw_boxes_on_image(raw_image):
    global boxes
    results = []
    boxes = detector.predict_one_page(raw_image)
    image = Image.fromarray(raw_image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min = min(point[0] for point in box)
        y_min = min(point[1] for point in box)
        x_max = max(point[0] for point in box)
        y_max = max(point[1] for point in box)
        draw.rectangle([x_min, y_min, x_max, y_max],
                       outline="red", width=2)

        patch = get_patch(raw_image, box)
        if recognizer:
            text = recognizer.predict_one_patch(patch).strip()
            results.append(text)
        else:
            results.append("<no recognizer>")

    return image, "\n".join(results)


def recognize_text_from_boxes(image):
    global boxes
    if not image or not boxes:
        return ""
    results = []
    for box in boxes:
        try:
            x_min = min(point[0] for point in box)
            y_min = min(point[1] for point in box)
            x_max = max(point[0] for point in box)
            y_max = max(point[1] for point in box)
            # cropped = image.crop((x_min, y_min, x_max, y_max))
            cropped = image[y_min:y_max, x_min:x_max]
            if recognizer:
                text = recognizer.predict_one_patch(cropped).strip()
                results.append(text)
            else:
                results.append("<no recognizer>")
        except Exception as e:
            results.append(f"[error] {e}")
    return "\n".join(results)


with gr.Blocks() as iface:
    gr.Markdown("# Hệ thống chú thích hình ảnh Hán - Nôm")

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("## 1. Phát hiện và nhận dạng văn bản")
            image_uploader = gr.Image(
                label="Ảnh gốc",
                type="numpy",
                height=250
            )
            process_image_button = gr.Button("📷 Dich ảnh 📷")
            processed_image_output = gr.Image(
                label="Phát hiện văn bản", type="pil", height=250)

        with gr.Column(scale=2):
            gr.Markdown("## 2. Dịch văn bản")
            source_text_area = gr.Textbox(
                lines=6,
                max_lines=6,
                label="Hán - Nôm",
                interactive=True
            )
            mode_dropdown = gr.Dropdown(
                choices=["Phiên âm", "Dịch nghĩa"],
                value="Phiên âm",
                label="Chế độ dịch",
                interactive=True
            )
            translate_button = gr.Button("⬇️ Dịch ⬇️")
            translated_text_area = gr.Textbox(
                lines=6,
                max_lines=6,
                label="Quốc Ngữ"
            )

    translate_button.click(
        fn=run_translation_model,
        inputs=[source_text_area, mode_dropdown],
        outputs=[translated_text_area]
    )

    process_image_button.click(
        fn=draw_boxes_on_image,
        inputs=[image_uploader],
        outputs=[processed_image_output, source_text_area]
    )

if __name__ == "__main__":
    load_all_model()
    iface.launch()
