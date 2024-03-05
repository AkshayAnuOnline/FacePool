#FacePool is an attendance system that utilizes facial recognition technology. 
#It offers features such as capturing images through a webcam, saving user details for training the facial recognition model, marking attendance in real-time, and displaying attendance data in a table format. 
#The system is developed using Python and integrates OpenCV for face detection, Tkinter for the graphical user interface, and pandas for data manipulation. FacePool provides a user-friendly interface for managing attendance records efficiently.
#Developed By Akshay Anu S 
#Linkedin - https://www.linkedin.com/in/akshayanu
#GitHub - https://github.com/AkshayAnuOnline


#Importing necessary libraries (Packages) and modules:
import tkinter as tk
from tkinter import * #for GUI
import sqlite3
from screeninfo import get_monitors #for retrieving monitor information
import cv2,os #cv2 for computer vision tasks (OpenCV)
import csv #for handling data
import numpy as np #for numerical computing
import pandas as pd #for handling data
import datetime
import time
from PIL import Image #Python library for image processing tasks
import glob #for searching files
from pandastable import Table #provides a table widget



#Setting up the GUI
window = Tk()
window.title("FacePool")
alphaerror = tk.PhotoImage(file = f"Asset\\alphaerror.png")
numerror = tk.PhotoImage(file = f"Asset\\numerror.png")
invalidentry = tk.PhotoImage(file = f"Asset\\invalidentry.png")
trained = tk.PhotoImage(file = f"Asset\\trained.png")
imagesaved = tk.PhotoImage(file = f"Asset\\imagesaved.png")
done = tk.PhotoImage(file = f"Asset\\done.png")
removed = tk.PhotoImage(file = f"Asset\\Removednoti.png")
deleteinvalid = tk.PhotoImage(file = f"Asset\\deleteinvalid.png")

conn = sqlite3.connect('FacePool.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS StudentDetails
             (Id INTEGER PRIMARY KEY, Name TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS TrainingDetails
             (ModelPath TEXT, Date TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS AttendanceDetails
             (FileName TEXT, Date TEXT)''')
conn.commit()
conn.close()
def is_number(s): #Checks if a string is a valid number
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False




#Captures images using the computer's webcam and saving them along with some user details for training the facial recognition model
def TakeImages():
    global alphaerror, numerror, trained, done, invalidentry
    Id = (entry0.get())
    name = (entry1.get())
    if (is_number(Id) and name.replace(" ", "").isalpha()):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\\" + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 75:
                break
        cam.release()
        cv2.destroyAllWindows()
        conn = sqlite3.connect('FacePool.db')
        c = conn.cursor()
        c.execute("INSERT INTO StudentDetails (Id, Name) VALUES (?, ?)", (Id, name))
        conn.commit()
        conn.close()
        c5 = canvas.create_image(293.0, 485.5, image=imagesaved)
        canvas.after(6000, lambda: canvas.itemconfig(c5, state='hidden'))
        TrainImages()
    else:
        if (is_number(Id)):
            c1 = canvas.create_image(293.0, 485.5, image=alphaerror)
            canvas.after(6000, lambda: canvas.itemconfig(c1, state='hidden'))
        if (name.replace(" ", "").isalpha()):
            c2 = canvas.create_image(293.0, 485.5, image=numerror)
            canvas.after(6000, lambda: canvas.itemconfig(c2, state='hidden'))
        if (name.strip() == "" and Id.strip() == ""):
            c3 = canvas.create_image(293.0, 485.5, image=invalidentry)
            canvas.after(6000, lambda: canvas.itemconfig(c3, state='hidden'))
        if (is_number(name) and Id.replace(" ", "").isalpha()):
            c4 = canvas.create_image(293.0, 485.5, image=invalidentry)
            canvas.after(6000, lambda: canvas.itemconfig(c4, state='hidden'))


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")

    conn = sqlite3.connect('FacePool.db')
    c = conn.cursor()
    c.execute("INSERT INTO TrainingDetails (ModelPath, Date) VALUES (?, ?)", ("TrainingImageLabel\Trainner.yml", datetime.datetime.now()))
    conn.commit()
    conn.close()

    c4 = canvas.create_image(293.0, 485.5, image=trained)
    canvas.after(6000, lambda: canvas.itemconfig(c4, state='hidden'))


#retrieves the images and corresponding labels from a specified directory
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty  list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids



def markattendance():

    extra_window2 = tk.Toplevel()
    WIN_WIDTH = 416
    WIN_HEIGHT = 500
    extra_window2.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window2.configure(bg="#FFFFFF")
    extra_window2.title("Mark Attendance")

    #performs real-time face recognition and attendance tracking using a trained model
    def TrackImages():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel\Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)

        conn = sqlite3.connect('FacePool.db')
        c = conn.cursor()
        c.execute("SELECT * FROM StudentDetails")
        df = pd.DataFrame(c.fetchall(), columns=['Id', 'Name'])

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        custom_name = entrycustomname.get().strip()  # Get custom name from entrycustomname
        if custom_name != "":
            fileName = f"Attendance\Attendance_{date}_{custom_name}.csv"
        else:
            fileName = f"Attendance\Attendance_{date}.csv"

        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 50:
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S %p')  # Include AM/PM in the timestamp
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id) + "-" + aa[0]  # Assuming aa is an array, select the first element which is the name
                    attendance.loc[len(attendance)] = [Id, aa[0], date, timeStamp]  # Use aa[0] to get the name without brackets
                else:
                    Id = ' '
                    tt = str(Id)
                if conf > 75:
                    noOfFile = len(os.listdir("ImagesUnknown")) + 1
                    cv2.imwrite(f"ImagesUnknown\Image{noOfFile}.jpg", im[y:y + h, x:x + w])
                cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)

            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            cv2.imshow('Face Recognition Window (Press q to close)', im)  # Set the name of the window
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):  # Close the window if 'q' or 'Q' is pressed
                break

        attendance.to_csv(fileName, mode='a', header=not os.path.exists(fileName), index=False)
        c.execute("INSERT INTO AttendanceDetails (FileName, Date) VALUES (?, ?)", (fileName, date))  # Insert file path into AttendanceDetails table
        cam.release()
        cv2.destroyAllWindows()
        c6 = canvas.create_image(207, 368, image=done)
        canvas.after(6000, lambda: canvas.itemconfig(c6, state='hidden'))
        conn.close()



    canvas = Canvas(
    extra_window2,
    bg = "#ffffff",
    height = 500,
    width = 416,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
    canvas.place(x = 0, y = 0)

    background_img = PhotoImage(file = f"Asset//markbg.png")
    background = canvas.create_image(
    208.0, 250.0,
    image=background_img)

    img0 = PhotoImage(file = f"Asset//markstart.png")
    b0 = Button(
    extra_window2,
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = TrackImages,
    activebackground="#0957A5",
    relief = "flat")

    b0.place(
    x = 152, y = 302,
    width = 110,
    height = 30)

    entry0_img = PhotoImage(file = f"Asset//markbox.png")
    entry0_bg = canvas.create_image(
    206.5, 264.0,
    image = entry0_img)

    entrycustomname = Entry(
    extra_window2,
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

    entrycustomname.place(
    x = 63.0, y = 247,
    width = 287.0,
    height = 32)

    extra_window2.iconbitmap(f"Asset\icon.ico")
    extra_window2.resizable(False, False)
    extra_window2.mainloop()



#retrieves attendance data from CSV files in a specific directory and displays it in a table format
import glob

def printdata():
    import glob

    # Path to the directory containing attendance CSV files
    attendance_dir = "Attendance/"

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(attendance_dir + "Attendance_*.csv")

    if not csv_files:
        print("No attendance files found.")
        return

    # Combine data from all CSV files into a single DataFrame
    all_data = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    all_data['Time'] = pd.to_datetime(all_data['Time'], format='%I:%M:%S %p')

    # Convert the 'Time' column to datetime format
    all_data['Time'] = pd.to_datetime(all_data['Time'])

    # Sort the data by 'Date' and 'Time' columns in ascending order
    all_data_sorted = all_data.sort_values(by=['Date', 'Time'])

    # Extract the time part from the datetime and convert to 12-hour format
    all_data_sorted['Time'] = all_data_sorted['Time'].dt.strftime('%I:%M:%S %p')

    # Display data using Pandas Table
    extra_window = tk.Toplevel()
    extra_window.title('Attendance')
    frame = tk.Frame(extra_window)
    frame.pack()

    pt = Table(frame, width=1100, height=618, dataframe=all_data_sorted, showtoolbar=True, showstatusbar=True)
    pt.show()

    WIN_WIDTH = 1100
    WIN_HEIGHT = 618
    extra_window.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window.iconbitmap(f"Asset\\icon.ico")
    extra_window.mainloop()



def printuserdata():
    conn = sqlite3.connect('FacePool.db')
    c = conn.cursor()

    c.execute("SELECT * FROM StudentDetails")
    df_users = pd.DataFrame(c.fetchall(), columns=['Id', 'Name'])

    # Display data using Pandas Table
    extra_window = tk.Toplevel()
    extra_window.title('Registered Users')
    frame = tk.Frame(extra_window)
    frame.pack()

    pt = Table(frame, width=1100, height=618, dataframe=df_users, showtoolbar=True, showstatusbar=True)
    pt.show()

    WIN_WIDTH = 1100
    WIN_HEIGHT = 618
    extra_window.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window.iconbitmap(f"Asset\icon.ico")
    extra_window.mainloop()

    conn.close()



#displays an "About FacePool" window
def about():
    extra_window2 = tk.Toplevel()
    WIN_WIDTH = 923
    WIN_HEIGHT = 518
    extra_window2.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window2.configure(bg="#FFFFFF")
    extra_window2.title("About FacePool")
    canvas1 = Canvas(
        extra_window2,
        bg="#FFFFFF",
        height=518,
        width=923,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas1.place(x=0, y=0)
    background_img1 = PhotoImage(file=f"Asset\\about.png")
    bg = canvas1.create_image(
        461.5, 259.5,
        image=background_img1)
    extra_window2.iconbitmap(f"Asset\icon.ico")
    extra_window2.resizable(False, False)
    extra_window2.mainloop()



def removeface():
    global removed, deleteinvalid
    def btn_clicked():
        student_id = entry0.get()
        if is_number(student_id):
            conn = sqlite3.connect('FacePool.db')
            c = conn.cursor()
            c.execute("DELETE FROM StudentDetails WHERE Id=?", (student_id,))
            conn.commit()
            conn.close()
            images = glob.glob(f"TrainingImage\\*.{student_id}.*.jpg")
            for image in images:
                os.remove(image)
            print(f"User with ID {student_id} deleted successfully.")
            c6 = canvas.create_image(209,380, image=removed)
            canvas.after(6000, lambda: canvas.itemconfig(c6, state='hidden'))
            TrainImages()
        else:
            c6 = canvas.create_image(211,380, image=deleteinvalid)
            canvas.after(6000, lambda: canvas.itemconfig(c6, state='hidden'))
            print("Invalid input. Please enter a valid ID.")


    def delete_all_students():
        global removed, deleteinvalid
        conn = sqlite3.connect('FacePool.db')
        c = conn.cursor()
        c.execute("SELECT * FROM StudentDetails")
        rows = c.fetchall()
        for row in rows:
            student_id = row[0]
            # Delete student record from the database
            c.execute("DELETE FROM StudentDetails WHERE Id=?", (student_id,))
            # Delete corresponding images from the TrainingImage directory
            images = glob.glob(f"TrainingImage/*.{student_id}.*.jpg")
            for image in images:
                os.remove(image)
        # Delete the training.yml file
        if os.path.exists("TrainingImageLabel/Trainner.yml"):
            os.remove("TrainingImageLabel/Trainner.yml")
        # Delete all images in the ImagesUnknown directory
        images_unknown = glob.glob("ImagesUnknown/*.jpg")
        for image in images_unknown:
            os.remove(image)
        c6 = canvas.create_image(209, 380, image=removed)
        canvas.after(6000, lambda: canvas.itemconfig(c6, state='hidden'))
        conn.commit()
        conn.close()




    extra_window2 = tk.Toplevel()
    WIN_WIDTH = 416
    WIN_HEIGHT = 415
    extra_window2.geometry(
        f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH) // 2}+{(get_monitors()[0].height - WIN_HEIGHT) // 2}")
    extra_window2.configure(bg="#FFFFFF")
    extra_window2.title("Delete User")
    canvas = Canvas(
        extra_window2,
        bg="#ffffff",
        height=415,
        width=416,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"Asset\\background123.png")
    background = canvas.create_image(
        208.0, 207.5,
        image=background_img)

    img0 = PhotoImage(file=f"Asset\\img110.png")
    b0 = Button(
        extra_window2,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        activebackground="#0957A5",
        relief="flat")
    b0.place(
        x = 153, y = 257,
    width = 110,
    height = 30)

    img1 = PhotoImage(file = f"Asset\\img112.png")
    b1 = Button(
    extra_window2,
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = delete_all_students,
    activebackground="#0957A5",
    relief = "flat")

    b1.place(
    x = 153, y = 311,
    width = 110,
    height = 30)

    entry0_img = PhotoImage(file=f"Asset\\textbox1.png")
    entry0_bg = canvas.create_image(
        207.5, 208.0,
        image=entry0_img)

    entry0 = Entry(
        extra_window2,
        bd=0,
        bg="#eaf4ff",
        highlightthickness=0)

    entry0.place(
        x=64.0, y=191,
        width=287.0,
        height=32)

    extra_window2.resizable(False, False)
    extra_window2.iconbitmap(f"Asset\icon.ico")
    extra_window2.mainloop()

def open_attendance_directory():
    # Get the current directory of the Python file
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # Create the path to the subfolder "Attendance"
    directory_path = os.path.join(current_directory, "Attendance")
    # Open the directory
    os.startfile(directory_path)
    

#GUI SETUP
window.iconbitmap(f"Asset\icon.ico")
WIN_WIDTH = 1100
WIN_HEIGHT = 618
window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{(get_monitors()[0].width - WIN_WIDTH)//2}+{(get_monitors()[0].height - WIN_HEIGHT)//2}")
window.configure(bg = "#ffffff")

canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 618,
    width = 1100,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"Asset\\background.png")
background = canvas.create_image(
    550.0, 315.0,
    image=background_img)

img0 = PhotoImage(file = f"Asset\img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = markattendance,
    relief = "flat")

b0.place(
    x = 735, y = 167,
    width = 210,
    height = 43)

img1 = PhotoImage(file = f"Asset\img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = printdata,
    relief = "flat")

b1.place(
    x = 735, y = 247,
    width = 210,
    height = 43)



img3 = PhotoImage(file = f"Asset\img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = TakeImages,
    activebackground="#0D5FB1",
    relief = "flat")

b3.place(
    x = 234, y = 382,
    width = 110,
    height = 30)

img10 = PhotoImage(file = f"Asset\img10.png")
b10 = Button(
    image = img10,
    borderwidth = 0,
    highlightthickness = 0,
    command = printuserdata,
    relief = "flat")

b10.place(
    x = 735, y = 327,
    width = 210,
    height = 43)

img11 = PhotoImage(file = f"Asset\img11.png")
b11 = Button(
    image = img11,
    borderwidth = 0,
    highlightthickness = 0,
    command = removeface,
    relief = "flat")

b11.place(
    x = 735, y = 407,
    width = 210,
    height = 43)

entry0_img = PhotoImage(file = f"Asset\img_textBox0.png")
entry0_bg = canvas.create_image(
    287.5, 235.0,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#eaf4ff",
    highlightthickness = 0)

entry0.place(
    x = 144.0, y = 218,
    width = 287.0,
    height = 32)

entry1_img = PhotoImage(file = f"Asset\img_textBox1.png")
entry1_bg = canvas.create_image(
    287.5, 320.0,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#eaf4ff",
    highlightthickness = 0)

entry1.place(
    x = 144.0, y = 303,
    width = 287.0,
    height = 32)

img4 = PhotoImage(file = f"Asset\img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = about,
    relief = "flat")

b4.place(
    x = 1046, y = 559,
    width = 37,
    height = 37)

img6 = PhotoImage(file = f"Asset//img6.png")
b6 = Button(
    image = img6,
    borderwidth = 0,
    highlightthickness = 0,
    command = open_attendance_directory,
    relief = "flat")

b6.place(
    x = 988, y = 556,
    width = 40,
    height = 43)

window.resizable(False, False)
window.mainloop()

#END OF CODE