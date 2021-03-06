import cv2
import tkinter as tk
import tkinter.filedialog
from tkinter.messagebox import showerror
from PIL import Image, ImageTk

class App(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.title("Facial Recognition | Choose File")
        self.minsize(width=300, height=50)
        font = ("Arial", 9)

        menu = tk.Menu(self)
        self.config(menu=menu)
        fileMenu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Open File", command=self.load_file)

        self.panel = tk.Label(self, image='')

    def load_file(self):
        fname = tk.filedialog.askopenfilename(filetypes=(('image files', '.png'), ('image files', '.jpg')))
        if fname:
            try:
                self.process(fname)
            except:
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return


    def process(self, img_path):
        imagePath = img_path
        cascPath = "haarcascade_frontalface_default.xml"

        faceCascade = cv2.CascadeClassifier(cascPath)

        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.35, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        print("Found {0} faces!".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        img = cv2.cvtColor(self.resizeKeepAR(img, width=300), cv2.COLOR_BGR2RGB)
        self.show_image(Image.fromarray(img))

    def show_image(self, img):
        imgtk = ImageTk.PhotoImage(image=img)

        self.panel.config(image='')
        self.panel.image = imgtk
        self.panel.config(image=imgtk)
        self.panel.pack(side="bottom", fill="both", expand="yes")

    def resizeKeepAR(self, image, width=None, height=None, interpolation=cv2.INTER_AREA):
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation)

if __name__ == '__main__':
    app = App(None)
    app.resizable(False, False)
    app.mainloop()