import numpy as np
import matplotlib.pyplot as plt

import zbar
import cv2
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

# uses zbar to scan an image for a QR code. Returns the four corners if found
def getQRCorners(imagepath):
    scanner = zbar.ImageScanner()

    # configure the reader
    scanner.parse_config('enable')

    # prepare image data
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    width, height = pil.size
    raw = pil.tobytes()

    # wrap image data
    zimage = zbar.Image(width, height, 'Y800', raw)

    # scan the image for barcodes
    scanner.scan(zimage)

    # extract results
    if zimage:
        for symbol in zimage:
            # return first qr location
            return np.array(symbol.location)
    # return 0 if failed
    return np.zeros((4,2))

# Displays corners on top of image
def QRDisplay(imagepath):
    locations = getQRCorners(imagepath)
    image = plt.imread(imagepath)
    plt.figure()
    plt.imshow(image)
    colors = ['ro','go','bo','yo']
    i=0
    for loc in locations:
        plt.plot(loc[0],loc[1],colors[i])
        i = i + 1
        
# Generates iphone6 intrinsic cam matrix
def iphone6_cm():
    width,height = 2448, 3264
    cx = width/2
    cy = height/2

    HFOV = 58.040
    VFOV = cy*HFOV/cx

    fx = abs(float(width) / (2 * np.tan(HFOV*np.pi/180 / 2)))
    fy = abs(float(height) / (2 * np.tan(VFOV*np.pi/180 / 2)))
    return np.array([[fx, 0, cx],[0,fy,cy],[0,0,1]])

# Generates camera "pose" in the form of a wireframe in the global coordinate system
def get_camera_wireframe(rvec, tvec, outwidth=4.0,outheight=3.0,zheight=4.0 ):
    inwidth = outwidth/2
    inheight = outheight/2

    cx = np.array([[-outwidth,  -outwidth,  -outwidth,  -outwidth ],
        [-outwidth, -inwidth, -inwidth, -outwidth],
        [ outwidth,  inwidth,  inwidth,  outwidth],
        [ outwidth,   outwidth,   outwidth,   outwidth ]])
    
    cy = np.array([[-outheight, -outheight,  outheight,  outheight],
        [-outheight, -inheight,  inheight,  outheight],
        [-outheight, -inheight,  inheight,  outheight],
        [-outheight, -outheight,  outheight,  outheight]])

    cz = np.array([[zheight,zheight,zheight,zheight],
                   [zheight,0,0,zheight],
                   [zheight,0,0,zheight],
                   [zheight,zheight,zheight,zheight]])

    pts = np.array([cx.flatten(),cy.flatten(),cz.flatten()])
    pts = cv2.Rodrigues(rvec)[0].dot(pts) + tvec
    pts = pts.reshape([3,4,4])
    return pts

# Displays camera pose based on input image path
def display_camera_pose(imagepath):
    # Assumes pattern.jpg is in certain location. 
    # uses this for visualization
    qr = plt.imread('images/pattern.jpg')/255.

    QRDisplay(imagepath)
    
    # P3P 
    corners = getQRCorners(imagepath)
    if not corners.any():
        print "Error. QR Code not found"
        return

    corners = corners.reshape((4,1,2)).astype(float)
    objectpoints = np.array([[-4.4,4.4,0],[-4.4,-4.4,0],[4.4,-4.4,0],[4.4,4.4,0]]).reshape(4,1,3)
    cameraMatrix = iphone6_cm()
    retval, rvec, tvec = cv2.solvePnP(objectpoints, corners, cameraMatrix,np.zeros((5,1)),flags=cv2.CV_P3P)
    
    # transform camera frame of reference
    R,J = cv2.Rodrigues(rvec)
    axislength = 8
    coords = np.array([[0,0,0],[axislength,0,0],[0,axislength,0],[0,0,axislength]]).T
    camcoords = np.dot(R,coords) + tvec

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot Image at center
    scale = 8.8/(288-42)
    X = np.arange(0,scale*330,scale)
    X = X - np.max(X)/2
    Y = X
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, 0,rstride=5, cstride=5, facecolors=qr)

    # Add Reference QR corner points
    colors = ['ro','go','bo','yo']
    i=0
    for objpt in objectpoints:
        ox,oy,oz = objpt.reshape([3,1])
        ax.plot(ox,oy,oz,colors[i])
        i = i+1

    # Add global coordinate frame
    ax.plot([0,6],[0,0],0,color='r')
    ax.plot([0,0],[0,6],0,color='g')
    ax.plot([0,0],[0,0],[0,6],color='b')

    # Add Camera frame
    camxaxis = camcoords[:,(0,1)]
    ax.plot(camxaxis[0],camxaxis[1],camxaxis[2],color='r')
    camyaxis = camcoords[:,(0,2)]
    ax.plot(camyaxis[0],camyaxis[1],camyaxis[2],color='g')
    camzaxis = camcoords[:,(0,3)]
    ax.plot(camzaxis[0],camzaxis[1],camzaxis[2],color='b')

    # Add Camera wireframe
    pts = get_camera_wireframe(rvec,tvec)
    ax.plot_wireframe(pts[0],pts[1],pts[2])


    # set x,y,z scale the same
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_zlim(0,60)

    plt.show()
