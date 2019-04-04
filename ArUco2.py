import numpy
import cv2
import cv2.aruco as aruco

id_aruco=2
num_pixels=400
def aruco_gen(id_aruco, num_pixels):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)     
                                                            
    img = aruco.drawMarker(aruco_dict, id_aruco, num_pixels)

   
    
    dst = cv2.copyMakeBorder(img, 25, 25, 25, 25,cv2.BORDER_CONSTANT,value=[255,0,0])
    
    img1=cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
  
    img1=cv2.putText(img1,'ArUco ID = 2',(120,21),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)    


    img1=img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    
    
    cv2.imshow('frame',img1)
    


    cv2.imwrite('Aruco2.jpeg',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


aruco_gen(id_aruco, num_pixels)
