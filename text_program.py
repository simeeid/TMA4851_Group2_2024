import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    text_widget.configure(state='normal')
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
            text_widget.delete('0.0', END)
            text_widget.insert(END, content)  # apply the correct color scheme after loading the file

def toggle_dark_mode():
    if dark_mode.get():
        root.configure(bg='black')
        # text_widget.configure(bg='black', fg='white', state='normal')
        ctk.set_appearance_mode('dark')
        # make dark_mode_switch also dark
        dark_mode_switch.configure(bg='black', fg='white', selectcolor='black')
    else:
        root.configure(bg='white')
        ctk.set_appearance_mode('light')
        dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')

root = Tk()
root.geometry('300x300')

text_widget = ctk.CTkTextbox(root)
text_widget.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.9)  # added relx, rely for left and right margins
text_widget.configure(wrap='word', state='disable')  # wrap around whole words

frame = ctk.CTkFrame(root, height=32, fg_color='red')
frame.place(relx=0.05, rely=0.925, relwidth=0.9)

load_button = ctk.CTkButton(frame, width=150, text="Load .txt file", command=load_file)
load_button.grid(row=0, column=0, sticky=ctk.N+ctk.S)  # place the button below the text widget

dark_mode = IntVar()
dark_mode_switch = Checkbutton(frame, text="Dark Mode", variable=dark_mode, command=toggle_dark_mode)
dark_mode_switch.grid(row=0, column=1, sticky=ctk.N+ctk.S)
if(ctk.get_appearance_mode().lower() == 'dark'):
    dark_mode.set(1)
    toggle_dark_mode()

root.mainloop()
