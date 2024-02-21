import time
import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk
import pyautogui

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
        # text_widget.configure(bg='black', fg='white', state='normal')
        ctk.set_appearance_mode('dark')
        # make dark_mode_switch also dark
        dark_mode_switch.configure(bg='black', fg='white', selectcolor='black')
        auto_scroll_switch.configure(bg='black', fg='white', selectcolor='black')
    else:
        root.configure(bg='white')
        ctk.set_appearance_mode('light')
        dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')
        auto_scroll_switch.configure(bg='white', fg='black', selectcolor='white')

scrolling = False  # Flag to check if auto_scroll is already running

def auto_scroll(textbox):
    """Scrolls the textbox at a constant speed."""
    global scrolling
    mouse_y = root.winfo_pointerxy()[1] / root.winfo_screenheight()
    # mouse_y = pyautogui.position()[1] / pyautogui.size()[1]
    if auto_scroll_var.get() and mouse_y > 0.75:
        textbox.yview_scroll(1, 'pixels')  # Scroll down by a small amount
        textbox.after(1, lambda: auto_scroll(textbox))  # Call itself after 1ms
    elif mouse_y < 0.25:
        textbox.yview_scroll(-1, 'pixels')  # Scroll up by a small amount
        textbox.after(1, lambda: auto_scroll(textbox))  # Call itself after 1ms
    else:
        scrolling = False  # Stop scrolling when mouse is not in the upper or lower quarter

def mouse_event(event):
    global scrolling
    y = event.y / root.winfo_height()
    if (y > 0.75 or y < 0.25) and not scrolling and auto_scroll_var.get():
        scrolling = True
        auto_scroll(text_widget)

root = Tk()
root.geometry('500x500')
root.minsize(500, 500)

text_widget = ctk.CTkTextbox(root)
text_widget.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.9)
text_widget.configure(wrap='word', state='disable')

frame = ctk.CTkFrame(root, height=32, fg_color='red')
frame.place(relx=0.05, rely=0.925, relwidth=0.9)

load_button = ctk.CTkButton(frame, width=150, text="Load .txt file", command=load_file)
load_button.grid(row=0, column=0, sticky=ctk.N+ctk.S)

dark_mode = IntVar()
dark_mode_switch = Checkbutton(frame, text="Dark Mode", variable=dark_mode, command=toggle_dark_mode)
dark_mode_switch.grid(row=0, column=1, sticky=ctk.N+ctk.S)

auto_scroll_var = IntVar()
auto_scroll_switch = Checkbutton(frame, text="Auto Scroll", variable=auto_scroll_var)
auto_scroll_switch.grid(row=0, column=2, sticky=ctk.N+ctk.S)

if(ctk.get_appearance_mode().lower() == 'dark'):
    dark_mode.set(1)
    toggle_dark_mode()

root.bind('<Motion>', mouse_event)

root.mainloop()
