from Modules.cnnModels import *
from Modules.Basic import *


#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", required = True, help = "path to the video file")
args = vars(parser.parse_args())
#==============================================================================
#==============================================================================
cap = cv2.VideoCapture(args['video'])
while (cap.isOpened()):
    plateRet, plateFrame = cap.read()

    #=========================================================================================
    #plate_Original, plate_morphEx, edge = preprocessOne(plateFrame, (42,10), True)
    plate_Original, plate_morphEx, edge = preprocessOne(plateFrame, (34,8), False)

    _,plateCountours,_ = cv2.findContours(plate_morphEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for plateCountour in plateCountours:
        aspect_ratio_range, area_range = (2.2, 12), (500, 18000)
        if validate_contour(plateCountour, plate_morphEx, aspect_ratio_range, area_range):
            rect = cv2.minAreaRect(plateCountour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(plate_Original, [box], 0, (0,255,0), 1) #change position after CNN
            Xs, Ys = [i[0] for i in box], [i[1] for i in box]
            x1, y1 = min(Xs), min(Ys)
            x2, y2 = max(Xs), max(Ys)

            angle = rect[2]
            if angle < -45: angle += 90

            W, H = rect[1][0], rect[1][1]
            aspect_ratio = float(W)/H if W > H else float(H)/W

            center = ((x1+x2)/2, (y1+y2)/2)
            size = (x2-x1, y2-y1)
            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
            tmp = cv2.getRectSubPix(edge, size, center)
            TmpW = H if H > W else W
            TmpH = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
            __,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            white_pixels = 0
            for x in range(tmp.shape[0]):
                for y in range(tmp.shape[1]):
                    if tmp[x][y] == 255:
                        white_pixels += 1

            edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

            tmp = cv2.getRectSubPix(plateFrame, size, center)
            tmp = cv2.warpAffine(tmp, M, size)
            TmpW = H if H > W else W
            TmpH = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(TmpW),int(TmpH)), (size[0]/2, size[1]/2))
            cv2.imshow("Tmp", plate_Original)

            #--------------------------------------------------------
            tmp = reshape(tmp, PLATE_IMG_SIZE, "plateBuffer.jpg")
            #--------------------------------------------------------
            data = tmp.reshape(PLATE_IMG_SIZE, PLATE_IMG_SIZE, 1)
            plate_model_out = plate_model.predict([data])[0]
            if not np.argmax(plate_model_out) == 1:
                cv2.drawContours(plate_Original, [box], 0, (0,0,255), 2) #change position after CNN
                charOrigin = copy.copy(tmp)
                charGaussian = cv2.GaussianBlur(tmp, (3,3), 0)
                charThresh = cv2.adaptiveThreshold(charGaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)
                #========================
                x, y, w, h = 0, 0, 0, 0
                charsBuffer = []
                #========================
                _,charContours,_ = cv2.findContours(charThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for charContour in charContours:
                    area = cv2.contourArea(charContour)
                    if area > 200 and area < 800:   [x,y,w,h] = cv2.boundingRect(charContour)
                    if h > 25 and h < 75 and w > 10 and w < 45:
                        if not len(charOrigin[y:y+h, x:x+w]) < 10:
                            Buffer = copy.copy(tmp[y:y+h, x:x+w])
                            cv2.rectangle(charOrigin, (x,y), (x+w,y+h), (255,0,0), 1)
                            charsBuffer.append([x, Buffer])
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                charsBuffer = sorted(charsBuffer, key= lambda x: x[0])
                TrueChars = []

                for i in range(len(charsBuffer)):
                    if i == 0:  TrueChars.append(charsBuffer[i])
                    elif charsBuffer[i][0] != charsBuffer[i-1][0]: TrueChars.append(charsBuffer[i])

                del (charsBuffer)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if len(TrueChars) >= 7 and len(TrueChars) <= 8:
                    cv2.imshow("[Plate] charContours", charOrigin)
                    #print("----------plate----------")
                    string_buffer = []
                    for i in range(len(TrueChars)):
                        Buffer = TrueChars[i][1]
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        Buffer = reshape(Buffer, CHARS_IMG_SIZE, "charBuffer.jpg")
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        data1 = Buffer.reshape(CHARS_IMG_SIZE, CHARS_IMG_SIZE, 1)
                        chars_model_out = chars_model.predict([data1])[0]
                        if not np.argmax(chars_model_out) == 36:
                            string_buffer.append(CodeToChar(chars_model_out))
                            pass
                        pass
                    if len(string_buffer) >= 7 and len(string_buffer) <= 8:
                        t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                        print("[{x}] ===> [{y}]".format(x=t, y=string_buffer))
                        pass
                    pass

                #
                #
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#=========================================================================================

cap.release()
cv2.destroyAllWindows()

os.system("rm charBuffer.jpg")
os.system("rm plateBuffer.jpg")
