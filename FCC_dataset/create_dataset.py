from pytube import YouTube
import csv, pdb,os,cv2,glob
import subprocess as sp
import dlib
import shutil


def ffmpeg(file):
    cmd  = ('ffmpeg -ss 00:00:00 -i '+'\"'+str(file)+'\"  -t 00:02:30 '+' -vf \"select=not(mod(n\,20))\" -vsync vfr -q:v 2 '+file[:-4]+'_%05d.jpg')
    sp.run(cmd, shell=True)

    cmd  = ('ffmpeg -sseof -180 -i '+'\"'+str(file)+'\"'+' -vf "select=not(mod(n\,20))" -vsync vfr -q:v 2 '+file[:-4]+'_eof_%05d.jpg')
    sp.run(cmd, shell=True)

def detect_face(path, file,form):
    img = dlib.load_rgb_image(file)
    dets = detector(img, 1)
    try:
        if len(dets)>0:
            for i, d in enumerate(dets[0:1]):
                img = img[max(0, d.top()): min(d.bottom(), img.shape[0]),
                                max(0, d.left()): min(d.right(), img.shape[1])]
            if img.shape[0] >400 and img.shape[1] >400:
                img2 = cv2.resize(img,(512,512))
                k = file.split("/")
                p = path+'HR/'+form+'/'
                f_path = p+k[-1][:-4]+"_512_cropped.jpg"
                if not os.path.exists(p):
                    os.makedirs(p)
                cv2.imwrite(f_path, cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))

            img = cv2.resize(img,(256,256))
            #win.clear_overlay()
            #win.set_image(img)
            k = file.split("/")
            p = path+'LR/'+form+'/'
            f_path = p+k[-1][:-4]+"_256_cropped.jpg"
            if not os.path.exists(p):
                os.makedirs(p)
        
            cv2.imwrite(f_path, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)




with open('./yt_frames_.csv', mode='r') as infile:
    reader = csv.reader(infile)
    urls = {rows[0]:[rows[1].split(';'),rows[2].split(';'),rows[3].split(';')] for rows in reader}


path = './Mkup_vids/'

if not os.path.exists(path):
    os.makedirs(path)

fcc_path = './FacialCosmeticContent/'
if not os.path.exists(fcc_path):
    os.makedirs(fcc_path+'base_frames')
    os.mkdir(fcc_path+'base_frames/nomakeup')
    os.mkdir(fcc_path+'base_frames/anymakeup')
    os.mkdir(fcc_path+'base_frames/strongmakeup')
    os.makedirs(fcc_path+'LR/NoMakeup')
    os.mkdir(fcc_path+'LR/AnyMakeup')
    os.mkdir(fcc_path+'LR/StrongMakeup')
    os.makedirs(fcc_path+'HR/NoMakeup')
    os.mkdir(fcc_path+'HR/AnyMakeup')
    os.mkdir(fcc_path+'HR/StrongMakeup')




detector = dlib.get_frontal_face_detector()


if(urls is not None):
    for a in urls.keys():
        try:
            url_ending = a
            a = 'https://www.youtube.com/watch?v='+a
            print(a)
            yt = YouTube(a)


            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not os.path.exists(path):
                os.makedirs(path)
            yt.download(path,url_ending)

            cap = cv2.VideoCapture(os.path.join(path,url_ending+'.mp4'))
            ffmpeg(os.path.join(path,url_ending+'.mp4'))
            frames = glob.glob(os.path.join(path,url_ending+'*.jpg'))
            print(urls[url_ending])

            for b in frames:
                if b.split(os.path.join(path,url_ending+'_'))[1][:-4] in urls[url_ending][0]:                    
                    detect_face(fcc_path,b,'NoMakeup')                   
                    os.rename(b,os.path.join(fcc_path,'base_frames/nomakeup',b.split('/')[-1])) 
                elif b.split(os.path.join(path,url_ending+'_'))[1][:-4] in urls[url_ending][2]:                    
                    detect_face(fcc_path,b,'StrongMakeup')
                    detect_face(fcc_path,b,'AnyMakeup')
                    os.rename(b,os.path.join(fcc_path,'base_frames/strongmakeup',b.split('/')[-1]))
                elif b.split(os.path.join(path,url_ending+'_'))[1][:-4] in urls[url_ending][1]:
                    detect_face(fcc_path,b,'AnyMakeup')              
                    os.rename(b,os.path.join(fcc_path,'base_frames/anymakeup',b.split('/')[-1]))                      
                else:
                    os.remove(b)
            os.remove(os.path.join(path,url_ending+'.mp4'))

        except Exception as e:
            print(e)
            print('Sadly {} is no longer available.'.format(a))
            continue
        # except:
        #     print('Download of {} failed'.format(a))
        #     continue

    shutil.rmtree(fcc_path+'base_frames')