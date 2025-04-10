from grounding_dino.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", "gdino_checkpoints/groundingdino_swint_ogc.pth")
IMAGE_PATH = "data/atm_libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo/images/demo_0/agentview_0.png"
TEXT_PROMPT = "the black bowl. the white plate."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)