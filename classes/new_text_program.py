import threading
from tracker import Tracker
import time
import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar, Radiobutton
from tkinter import Tk
from gaze import Gaze

# Shared data structure for communication, e.g., a list, dict, Queue, etc.
global shared_data_tracker
global shared_data_gaze
global text_widget
global root
global tracker_scroll_var
global gaze_scroll_var
global tracker_face
global tracker_gaze
global thread_gaze
global thread_face
global scrolling
shared_data_tracker = 0
shared_data_gaze = 0

def auto_scroll(textbox):
    """Scrolls the textbox at a constant speed."""
    global scrolling
    global tracker

    mouse_y = root.winfo_pointerxy()[1] / root.winfo_screenheight()
    # mouse_y = pyautogui.position()[1] / pyautogui.size()[1]

    print("tracker_scroll_var", tracker_scroll_var.get())

    if tracker_scroll_var.get() == 1 and tracker_face.shared_data < -2:
        textbox.yview_scroll(1, 'pixels')  # Scroll down by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif tracker_scroll_var.get() == 1 and tracker_face.shared_data > 20:
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif tracker_scroll_var.get() == 2 and tracker_gaze.detection == 2:
        print("gaze up")
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif tracker_scroll_var.get() == 2 and tracker_gaze.detection == 1:
        print("gaze down")
        textbox.yview_scroll(1, 'pixels')  # Scroll down by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    else:
        scrolling = False  # Stop scrolling when mouse is not in the upper or lower quarter

def run_app():
    def load_file():
        text_widget.configure(state='normal')
        file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                text_widget.delete('0.0', END)
                text_widget.insert(END, content)  # apply the correct color scheme after loading the file
        text_widget.configure(state='disable')

    def toggle_dark_mode():
        if dark_mode.get():
            root.configure(bg='black')
            ctk.set_appearance_mode('dark')
            # make dark_mode_switch also dark
            dark_mode_switch.configure(bg='black', fg='white', selectcolor='black')
            tracker_scroll_switch.configure(bg='black', fg='white', selectcolor='black')
            gaze_scroll_switch.configure(bg='black', fg='white', selectcolor='black')
            off_switch.configure(bg='black', fg='white', selectcolor='black')

            frame.configure(fg_color='black')
        else:
            root.configure(bg='white')
            ctk.set_appearance_mode('light')
            dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')
            tracker_scroll_switch.configure(bg='white', fg='black', selectcolor='white')
            gaze_scroll_switch.configure(bg='white', fg='black', selectcolor='white')
            frame.configure(fg_color='white')

    def start_gaze():
        global tracker_gaze
        global gaze_running
        global tracker_running
        
        tracker_face.stop_camera()
        tracker_gaze.running = True

    def start_face():
        global tracker_face
        global gaze_running
        global tracker_running

        tracker_gaze.stop_camera()
        tracker_face.running = True
        

    def stop_tracking():
        global tracker_face
        global tracker_gaze
        global gaze_running
        global tracker_running

        tracker_face.stop_camera()
        tracker_gaze.stop_camera()

    global scrolling # = False  # Flag to check if auto_scroll is already running
    scrolling = False


    def mouse_event(event):
        global scrolling
        y = event.y / root.winfo_height()
        if (y > 0.75 or y < 0.25) and not scrolling and tracker_scroll_var.get():
            scrolling = True
            auto_scroll(text_widget)

    global root
    root = Tk()
    root.geometry('500x500')
    root.minsize(500, 500)

    global text_widget
    text_widget = ctk.CTkTextbox(root)
    text_widget.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.9)
    text_widget.configure(wrap='word', state='disable')

    frame = ctk.CTkFrame(root, height=32)
    frame.place(relx=0.05, rely=0.925, relwidth=0.9)

    load_button = ctk.CTkButton(frame, width=150, text="Load .txt file", command=load_file)
    load_button.grid(row=0, column=0, sticky=ctk.N+ctk.S)

    dark_mode = IntVar()
    dark_mode_switch = Checkbutton(frame, text="Dark Mode", variable=dark_mode, command=toggle_dark_mode)
    dark_mode_switch.grid(row=0, column=1, sticky=ctk.N+ctk.S)

    

    global tracker_scroll_var
    # tracker_scroll_var = IntVar()
    # tracker_scroll_switch = Checkbutton(frame, text="Head tracking scroll", variable=tracker_scroll_var, command=start_tracking)
    # tracker_scroll_switch.grid(row=0, column=2, sticky=ctk.N+ctk.S)

    tracker_scroll_var = IntVar()
    off_switch = Radiobutton(frame, text="Off", variable=tracker_scroll_var, value=0, command=stop_tracking)
    off_switch.grid(row=0, column=2, sticky=ctk.N+ctk.S)

    tracker_scroll_switch = Radiobutton(frame, text="Head tracking scroll", variable=tracker_scroll_var, value=1, command=start_face)
    tracker_scroll_switch.grid(row=0, column=3, sticky=ctk.N+ctk.S)

    #global gaze_scroll_var
    # gaze_scroll_var = IntVar()
    # gaze_scroll_switch = Checkbutton(frame, text="Eye gaze scroll", variable=gaze_scroll_var, command=start_gaze)
    # gaze_scroll_switch.grid(row=0, column=3, sticky=ctk.N+ctk.S)

    #gaze_scroll_var = IntVar()
    gaze_scroll_switch = Radiobutton(frame, text="Eye gaze scroll", variable=tracker_scroll_var, value=2, command=start_gaze)
    gaze_scroll_switch.grid(row=0, column=4, sticky=ctk.N+ctk.S)

    if(ctk.get_appearance_mode().lower() == 'dark'):
        dark_mode.set(1)
        toggle_dark_mode()

    root.mainloop()
    stop_tracking()

def run_face_tracking():
    global tracker_face
    print("testface")
    tracker_face = Tracker(shared_data_tracker)
    tracker_face.start_camera()

    while True:

        if tracker_face.running:
            tracker_face.start_video()

        time.sleep(0.25)
        
    
def run_gaze_tracking():
    global tracker_gaze
    print("testgaze")
    tracker_gaze = Gaze(shared_data_gaze)
    tracker_gaze.start_camera()

    while True:

        if tracker_gaze.running:
            tracker_gaze.start_video()

        time.sleep(0.25)
        

def run_sampling():
    global text_widget
    global scrolling

    # global shared_data_tracker
    # global shared_data_gaze
    global tracker_face
    global tracker_gaze
    
    while True:

        if not scrolling:
            scrolling = True
            auto_scroll(text_widget)

        time.sleep(0.25)

        # print("gaze", shared_data_gaze)
        print("gaze", tracker_gaze.detection)
        # print("face", shared_data_tracker)
        print("face", tracker_face.shared_data)


# Create threads
thread_app = threading.Thread(target=run_app)
thread_sampling = threading.Thread(target=run_sampling)
thread_gaze = threading.Thread(target=run_gaze_tracking)
thread_face = threading.Thread(target=run_face_tracking)

# Start threads / text application
thread_app.start()
thread_gaze.start()
thread_face.start()
time.sleep(4)
thread_sampling.start()

# Wait for threads to complete
thread_app.join()
thread_face.join()
thread_gaze.join()
thread_sampling.join()
