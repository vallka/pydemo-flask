from flask import Flask, render_template, request
from werkzeug import secure_filename

import cv2
import numpy as np
import os

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html",cwd=os.getcwd())

@app.route('/about')
def about():
    return render_template("about.html",cwd=os.getcwd(),file=__file__,dir=os.path.realpath(__file__))

@app.route('/listcolors')
def listcolors():
    cc = get_color_samples()
    cc2 = create_plain_colors(cc)
    return render_template("listcolors.html",colors=cc)

@app.route('/instagram', methods = ['GET', 'POST'])
def instagram():
    if request.method == 'POST':
        from urllib.request import urlopen
        import re


        url = request.form['file']
        response = urlopen(url)
        s = response.read()

        fname = url.replace('https://www.instagram.com/p/','')
        fname = fname.replace('/','')
        filename = 'static/uploads/' + secure_filename(fname) + '.jpg'

        pa = re.compile('<meta property="og:image" content="([^>]+)"')
        found = pa.search(str(s))
        ff = found.group(1)

        response = urlopen(ff)
        

        f = open(filename, "wb")
        f.write(response.read()) 
        f.close()

        ret = process_file(filename)
        if ret[1]=='':
            return render_template("instagram.html",filename=filename,msg=ret[0])

        return render_template("instagram.html",filename=filename,msg=ret[0],file1=ret[1],file2=ret[2],file3=ret[3],file4=ret[4],file5=ret[5])
    
    return render_template("instagram.html")

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        filename = 'static/uploads/' + secure_filename(f.filename)
        f.save(filename)

        ret = process_file(filename)

        if ret[1]=='':
            return render_template("upload.html",filename=filename,msg=ret[0])

        return render_template("upload.html",filename=filename,msg=ret[0],file1=ret[1],file2=ret[2],file3=ret[3],file4=ret[4],file5=ret[5])
    
    return render_template("upload.html")

def process_file(filename):
    path,file = os.path.split(filename)

    text = ''
    #text = path + '*' +file+' |'

    img = cv2.imread(filename)  
    faces_detected_img,x,y,w,h,y0,h0 = detect_faces(img,1.1)  

    if x<0:
        return ['Hair not detected - please use another image','','','','','']


    x=int(x-w/8)
    print (x)
    x=max(x,0)
    print (x)
    w = int(w*1.25)

    rect = (x,y,w,h+h0)


    cv2.rectangle(faces_detected_img, (x, y), (x+w, y+h+h0), (0, 255, 0), 2)             


    #cv2.rectangle(faces_detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)             

    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img.copy(),mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    img2 = img * mask2[:,:,np.newaxis]

    img2[np.where((img2==[0,0,0]).all(axis=2))] = [255,255,255];

    hair = img2[y:y+h,x:x+w].copy()



    faces_detected_filename = 'static/created/'+file
    hair_filename = 'static/created/hair_'+file
    cv2.imwrite(faces_detected_filename,faces_detected_img)  
    cv2.imwrite(hair_filename,hair)  

    data = np.reshape(hair, (-1,3))
    #print(data.shape)
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,3,None,criteria,10,flags)

    #text += ' Dominant color is: bgr({})'.format(centers[0].astype(np.int32))
    #text += ' bgr({})'.format(centers[1].astype(np.int32))
    #text += ' bgr({})'.format(centers[2].astype(np.int32))

    res = centers[labels.flatten()]
    hair2 = res.reshape((hair.shape))
    hair2_filename = 'static/created/hair2_'+file
    cv2.imwrite(hair2_filename,hair2)  

    y=labels.flatten()
    c = [np.count_nonzero(y == 0),np.count_nonzero(y == 1),np.count_nonzero(y == 2)]

    #text += str(c[0]) +'/'+ str(c[1]) +'/'+ str(c[2])

    mc = -1
    for i in  range(0,3):
        #text += '*'+str(i)+'*'
        if centers[i][0]>=250 and centers[i][1]>=250 and centers[i][2]>=250:
            continue
        if mc<0 or c[i]>c[mc]:
            mc = i
    

    img = np.ones((53,200,3),np.uint8)
    img[:,:,0] = centers[mc][0]
    img[:,:,1] = centers[mc][1]
    img[:,:,2] = centers[mc][2]

    color_filename = 'static/created/color_'+file
    cv2.imwrite(color_filename,img)  

    color = centers[mc]
    cc = get_color_samples()
    closest=[999,'']
    for c in cc:
        ab = abs(c[1][0]-color[0]) + abs(c[1][1]-color[1]) + abs(c[1][2]-color[2])
        if ab<closest[0]:
            closest[0] = ab
            closest[1] = c[0]
            closest_color = c[1]


    #text += closest_color
    text += ' Main color is: bgr({}) - bgr({})'.format(centers[mc].astype(np.int32),closest_color)

    return [text,faces_detected_filename,hair_filename,hair2_filename,color_filename,closest[1]]


def detect_faces(colored_img, scaleFactor = 1.2):
    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    f_cascade = haar_face_cascade

    img_copy = colored_img.copy()          
 
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
 
    #let's detect multiscale (some images may be closer to camera than others) images
    
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

    if len(faces) != 1:
        return [img_copy,-1,-1,-1,-1,-1,-1]

    #go over list of faces and draw them as rectangles on original colored img

    for (x, y, w, h) in faces[0:1]:
        y2 = max(int(y-h/2.5) , 0)
        #crop_img = img_copy[y2:y,x:x+w].copy()
        #cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              
        #cv2.rectangle(img_copy, (x, y), (x+w, y2), (0, 0, 255), 2)              
        return [img_copy,x,y2,w,y-y2,y,h]

    return [img_copy,-1,-1,-1,-1,-1,-1]


def get_color_samples():
    from pathlib import Path

    p = Path('static/colors')
    pathlist=p.glob('**/*.jpg')

    colorsamples = []
    for path in pathlist:
        path_in_str = str(path)
        print (path.name)
        test = cv2.imread(path_in_str)  
        #cv2.imwrite(path_in_str+'.jpg',test)
        data = np.reshape(test, (-1,3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
        
        
        #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
        #print (int(centers[0][0]),int(centers[0][1]),int(centers[0][2]))
        
        colorsample = [path.name,[int(centers[0][0]),int(centers[0][1]),int(centers[0][2])]]
        
        #print (colorsample)
        colorsamples.append(colorsample)
        

    return colorsamples

def create_plain_colors(cc):
    for c in cc:
        plain_filename = 'static/plain_colors/'+c[0]
        img = np.ones((53,200,3),np.uint8)
        img[:,:,0] = c[1][0]
        img[:,:,1] = c[1][1]
        img[:,:,2] = c[1][2]

        cv2.imwrite(plain_filename,img)  




##########################
if __name__=="__main__":
    app.run(debug=True)

