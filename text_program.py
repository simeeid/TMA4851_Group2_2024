import time
import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk

def load_file():
    text_widget.configure(state='normal')
    # empty the text widget
    # text_widget.delete('0.0', END)
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
            text_widget.delete('0.0', END)
            text_widget.insert(END, content)  # apply the correct color scheme after loading the file
    text_widget.configure(state='disable')
    auto_scroll(text_widget, 0)

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

def auto_scroll(textbox, x):
    """Scrolls the textbox at a constant speed."""
    textbox.yview_moveto(x)  # Scrolls to the end of the textbox

    num_lines = textbox.get('1.0', END).count('\n')
    # find the width of the window
    window_width = textbox.winfo_width()

    x = x + 0.00004 / num_lines * window_width
    textbox.after(10, lambda: auto_scroll(textbox, x))


    # find the height of the text document or the number of lines, then update x proportionally
    #x = 1 - (textbox.yview()[1] - textbox.yview()[0])  # 1 - (bottom - top)
    # get the height of the text widget

    # find the number of line changes in the text widget
    # print(num_line_breaks)

    # textbox.after(10, lambda: auto_scroll(textbox, x))  # Calls itself every 100ms

    # textbox.yview_scroll(1, 'units')  # scroll 10% of the height of the text widget
    # if x%10 == 0:
    #     textbox.after(1000, lambda: auto_scroll(textbox, x+1))
    # else:
    #     textbox.after(100, lambda: auto_scroll(textbox, x+1))

root = Tk()
root.geometry('500x500')
root.minsize(500, 500)

text_widget = ctk.CTkTextbox(root)
text_widget.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.9)
text_widget.configure(wrap='word', state='disable')

frame = ctk.CTkFrame(root, height=32, fg_color='red')
frame.place(relx=0.05, rely=0.925, relwidth=0.9)

# upper_widget = ctk.CTkFrame(root, fg_color='blue')
# upper_widget.place(relx=0.0, rely=0.0, relheight=0.25, relwidth=1)

# make lower transparent, alpha=0.5 is not a valid argument
# lower_widget = ctk.CTkFrame(root)
# lower_widget.place(relx=0.0, rely=0.75, relheight=0.25, relwidth=1)

load_button = ctk.CTkButton(frame, width=150, text="Load .txt file", command=load_file)
load_button.grid(row=0, column=0, sticky=ctk.N+ctk.S)

dark_mode = IntVar()
dark_mode_switch = Checkbutton(frame, text="Dark Mode", variable=dark_mode, command=toggle_dark_mode)
dark_mode_switch.grid(row=0, column=1, sticky=ctk.N+ctk.S)
if(ctk.get_appearance_mode().lower() == 'dark'):
    dark_mode.set(1)
    toggle_dark_mode()

root.mainloop()
