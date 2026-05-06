import os
import numpy as np
import logging
from io import BytesIO

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
MODEL_PATH = os.environ.get("MODEL_PATH", "fruit_model.h5")

CLASSES = [
    "🍎 Fresh Apple",
    "🍌 Fresh Banana",
    "🍊 Fresh Orange",
    "🍎 Rotten Apple",
    "🍌 Rotten Banana",
    "🍊 Rotten Orange",
]

# ─── Load model once at startup ────────────────────────────────────────────────
logger.info("Loading model from %s ...", MODEL_PATH)
model = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def predict_image(img_bytes: bytes) -> tuple[str, float]:
    """Return (class_name, confidence) for raw image bytes."""
    img = image.load_img(BytesIO(img_bytes), target_size=(100, 100))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])


# ─── Handlers ──────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I'm a *Fruit Freshness Bot*.\n\n"
        "Send me a photo of an apple 🍎, banana 🍌, or orange 🍊 "
        "and I'll tell you whether it's *fresh or rotten*!",
        parse_mode="Markdown",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 *How to use:*\n"
        "1. Take a photo of a fruit\n"
        "2. Send it to this chat\n"
        "3. I'll predict: Fresh or Rotten\n\n"
        "Supported fruits: Apple, Banana, Orange.",
        parse_mode="Markdown",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Analyzing your fruit photo...")

    # Download the highest-resolution version of the photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    img_bytes = await file.download_as_bytearray()

    try:
        label, confidence = predict_image(bytes(img_bytes))
        freshness = "✅ FRESH" if "Fresh" in label else "❌ ROTTEN"
        await update.message.reply_text(
            f"*Result:* {label}\n"
            f"*Status:* {freshness}\n"
            f"*Confidence:* {confidence * 100:.1f}%",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("Prediction error")
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't process that image. "
            "Please send a clear photo of a fruit."
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allow users to send images as files (documents)."""
    doc = update.message.document
    if doc.mime_type and doc.mime_type.startswith("image/"):
        file = await context.bot.get_file(doc.file_id)
        img_bytes = await file.download_as_bytearray()
        try:
            label, confidence = predict_image(bytes(img_bytes))
            freshness = "✅ FRESH" if "Fresh" in label else "❌ ROTTEN"
            await update.message.reply_text(
                f"*Result:* {label}\n"
                f"*Status:* {freshness}\n"
                f"*Confidence:* {confidence * 100:.1f}%",
                parse_mode="Markdown",
            )
        except Exception:
            await update.message.reply_text("⚠️ Could not process this image file.")
    else:
        await update.message.reply_text("Please send an image file.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Please send me a *photo* of a fruit 🍎🍌🍊",
        parse_mode="Markdown",
    )


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    app = ApplicationBuilder().token("8764077989:AAGamQne3I5qi6VgTckywFF7n_8qaRBV13Q").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
