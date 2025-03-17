import wandb
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def logClassImages(project_name:str):
  wandb.init(project=project_name)
  wandb_image_indices = []

  for classNumber in range(10):
    for j in range(len(y_test)):
      if y_test[j] == classNumber:
        wandb_image_indices.append(x_test[j])
        break

  wandb_images = [wandb.Image(wandb_image_indices[i], caption = classes[i]) for i in range(10)]
  wandb.log({"Sample images for each class": wandb_images})
  wandb.finish()

logClassImages("da6401_assignment1")
