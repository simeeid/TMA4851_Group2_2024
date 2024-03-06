import threading
from tracker import Tracker
import time
import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk

# Shared data structure for communication, e.g., a list, dict, Queue, etc.
global shared_data
global text_widget
global root
global auto_scroll_var
global tracker
global scrolling
shared_data = 0

def auto_scroll(textbox):
    """Scrolls the textbox at a constant speed."""
    global scrolling
    global tracker
    # y_value = tracker.shared_data / 25

    mouse_y = root.winfo_pointerxy()[1] / root.winfo_screenheight()
    # mouse_y = pyautogui.position()[1] / pyautogui.size()[1]

    if auto_scroll_var.get() and tracker.shared_data < -2:
        textbox.yview_scroll(1, 'pixels')  # Scroll down by a small amount
        textbox.after(5, lambda: auto_scroll(textbox))  # Call itself after 1ms

    elif auto_scroll_var.get() and tracker.shared_data > 20:
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
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
            auto_scroll_switch.configure(bg='black', fg='white', selectcolor='black')
            frame.configure(fg_color='black')
        else:
            root.configure(bg='white')
            ctk.set_appearance_mode('light')
            dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')
            auto_scroll_switch.configure(bg='white', fg='black', selectcolor='white')
            frame.configure(fg_color='white')

    global scrolling # = False  # Flag to check if auto_scroll is already running
    scrolling = False


    def mouse_event(event):
        global scrolling
        y = event.y / root.winfo_height()
        if (y > 0.75 or y < 0.25) and not scrolling and auto_scroll_var.get():
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

    global auto_scroll_var
    auto_scroll_var = IntVar()
    auto_scroll_switch = Checkbutton(frame, text="Auto Scroll", variable=auto_scroll_var)
    auto_scroll_switch.grid(row=0, column=2, sticky=ctk.N+ctk.S)

    if(ctk.get_appearance_mode().lower() == 'dark'):
        dark_mode.set(1)
        toggle_dark_mode()

    # root.bind('<Motion>', mouse_event)

    root.mainloop()
    tracker.running = False

def run_face_tracking():
    global tracker
    tracker = Tracker(shared_data)
    tracker.start_video()
    
def run_sampling():
    global text_widget
    global shared_data
    global tracker
    global scrolling
    while tracker.running:

        if not scrolling:
            scrolling = True
            auto_scroll(text_widget)

        time.sleep(0.25)


# Create threads
thread1 = threading.Thread(target=run_app)
thread2 = threading.Thread(target=run_face_tracking)
thread3 = threading.Thread(target=run_sampling)

# Start threads
thread1.start()
thread2.start()
time.sleep(4)
thread3.start()

# Wait for both threads to complete
thread1.join()
thread2.join()
thread3.join()
