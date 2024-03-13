import threading
from tracker import Tracker
import time
import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk
from gaze import Gaze

# Shared data structure for communication, e.g., a list, dict, Queue, etc.
global shared_data_tracker
global shared_data_gaze
global text_widget
global root
global tracker_scroll_var
global gaze_scroll_var
global tracker
global scrolling
global thread2
global gaze_running
global tracker_running
shared_data_tracker = 0
shared_data_gaze = 0
gaze_running = False
tracker_running = False
tracker = Gaze(0)

def auto_scroll(textbox):
    """Scrolls the textbox at a constant speed."""
    global scrolling
    global tracker

    mouse_y = root.winfo_pointerxy()[1] / root.winfo_screenheight()
    # mouse_y = pyautogui.position()[1] / pyautogui.size()[1]

    if tracker_scroll_var.get() and tracker.shared_data_tracker < -2:
        textbox.yview_scroll(1, 'pixels')  # Scroll down by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif tracker_scroll_var.get() and tracker.shared_data_tracker > 20:
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif gaze_scroll_var.get() and shared_data_gaze == 2:
        print("gaze up")
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif gaze_scroll_var.get() and shared_data_gaze == 1:
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
            frame.configure(fg_color='black')
        else:
            root.configure(bg='white')
            ctk.set_appearance_mode('light')
            dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')
            tracker_scroll_switch.configure(bg='white', fg='black', selectcolor='white')
            gaze_scroll_switch.configure(bg='white', fg='black', selectcolor='white')
            frame.configure(fg_color='white')

    def start_gaze():
        global tracker
        global gaze_running
        global tracker_running
        if not gaze_running:
            if not tracker_running and not tracker.running:
                print("Starting gaze tracker")
                run_tracker("gaze")
            else:
                print("Face tracker is already runnin, please stop face tracker first.")
            gaze_running = True
        else:
            if not tracker_running and tracker.running:
                print("Stopping gaze since it is already running")
                tracker.running = False

            gaze_running = False

    def start_tracking():
        global tracker
        global gaze_running
        global tracker_running
        if not tracker_running:
            if not gaze_running and not tracker.running:
                print("Starting face tracker")
                run_tracker("face")
            else:
                print("Gaze tracker is already runnin, please stop gaze tracker first.")
            tracker_running = True
        else:
            if not gaze_running and tracker.running:
                print("Stopping face tracker since it is already running")
                tracker.running = False

            tracker_running = False

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
    tracker_scroll_var = IntVar()
    tracker_scroll_switch = Checkbutton(frame, text="Head tracking scroll", variable=tracker_scroll_var, command=start_tracking)
    tracker_scroll_switch.grid(row=0, column=2, sticky=ctk.N+ctk.S)

    global gaze_scroll_var
    gaze_scroll_var = IntVar()
    gaze_scroll_switch = Checkbutton(frame, text="Eye gaze scroll", variable=gaze_scroll_var, command=start_gaze)
    gaze_scroll_switch.grid(row=0, column=3, sticky=ctk.N+ctk.S)

    if(ctk.get_appearance_mode().lower() == 'dark'):
        dark_mode.set(1)
        toggle_dark_mode()

    # root.bind('<Motion>', mouse_event)

    root.mainloop()
    tracker.running = False

def run_face_tracking():
    global tracker
    tracker = Tracker(shared_data_tracker)
    tracker.start_video()
    
def run_gaze_tracking():
    global tracker
    tracker = Gaze(shared_data_gaze)
    tracker.start_video()

def run_tracker(type):
    global thread2

    if type == "gaze":
        thread2 = threading.Thread(target=run_gaze_tracking)
        thread2.start()
        time.sleep(4)
        thread3.start()

    elif type == "face":
        thread2 = threading.Thread(target=run_face_tracking)
        thread2.start()
        time.sleep(4)
        thread3.start()

def run_sampling():
    global text_widget
    global shared_data_tracker
    global shared_data_gaze
    global tracker
    global scrolling
    while tracker.running:

        if not scrolling:
            scrolling = True
            auto_scroll(text_widget)

        time.sleep(0.25)


# Create threads
thread1 = threading.Thread(target=run_app)
# thread2 = threading.Thread(target=run_tracker("gaze"))
thread3 = threading.Thread(target=run_sampling)

# Start threads
thread1.start()
# thread2.start()
# time.sleep(4)
# thread3.start()

# Wait for both threads to complete
thread1.join()
thread2.join()
thread3.join()
