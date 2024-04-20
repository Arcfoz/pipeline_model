import tensorflow as tf
import rantea_func

model_path = "1.Object Detection.onnx"
loaded_best_model = tf.keras.models.load_model("2.Image Recognition.h5")

pathimg = "BT3.jpg"

cropped_image = rantea_func.object_detection(image_path=pathimg, model_path=model_path)
prediction = rantea_func.predict(img=cropped_image, loaded_best_model=loaded_best_model) # Output
print(prediction)