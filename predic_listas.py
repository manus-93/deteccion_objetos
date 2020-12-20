# predic_json.py


import cv2
import matplotlib.pyplot as plt
import numpy as np

# parameters
image_path = 'imgs/img00012.jpg'
confidence_img = .4
threshold_score = .3
threshold__non_maximum_suppression = .3



# loading model
print("Cargando modelo...")
net = cv2.dnn.readNetFromDarknet('yolov3_custom.cfg','yolov3_custom_09_12.weights')
print("Modelo cargado!")
classes = ['Goggles','Mask','Helmet']

# read image in format .jpg
my_img = cv2.imread(image_path)


def v_pred(my_img):
    # funcion v_pred es la que reliza la deteccion de objetos en las clases establecidas
    # input: imagen en formato jpg
    # output: una lista con los objetos DETECTADOS en la donde cada elemento de la lista
    # tiene la forma: [id_img, label, confidence, box_detect[x, x+w, y, y+h]]
    
    # shape de la imagen imagen
    ht,wt,_ = my_img.shape
    
    # realizo deteccion de objetos
    blop = cv2.dnn.blobFromImage(my_img,1/255,(416,416),swapRB = True,crop = False)
    net.setInput(blop)
    last_layer = net.getUnconnectedOutLayersNames()
    layer_out = net.forward(last_layer)
    
    # todas las detecciones posibles en la imagen que cumplen con la confiabilidad establecida
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > confidence_img:         # confidence
                center_x = int(detection[0]*wt)
                center_y = int(detection[1]*ht)
                w = int(detection[2]*wt)
                h = int(detection[3]*ht)

                x= int(center_x - w/2)
                y= int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    # Aplico el non maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, threshold_score, threshold__non_maximum_suppression)

    # preparo el output
    out = []
    if len(indexes) == 0:
        return(["No detecta nada de nada"])
    else:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i],3)
            out.append([class_ids[i],label,confidence,[x,x+w,y,y+h]])
        return(out)


print(v_pred(my_img))