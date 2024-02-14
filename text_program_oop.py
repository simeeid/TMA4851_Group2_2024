import customtkinter as ctk
from tkinter import filedialog, END, Checkbutton, IntVar
from tkinter import Tk

class BasicView():

    def __init__(self):
        self.root = Tk()
        # self.text_widget = 
        self.root.geometry('500x500')
        self.root.minsize(500, 500)

        self.text_widget = ctk.CTkTextbox(self.root)
        self.text_widget.place(relx=0.05, rely=0.025, relwidth=0.9, relheight=0.9)
        self.text_widget.configure(wrap='word', state='disable')

        frame = ctk.CTkFrame(self.root, height=32, fg_color='red')
        frame.place(relx=0.05, rely=0.925, relwidth=0.9)

        load_button = ctk.CTkButton(frame, width=150, text="Load .txt file", command=self.load_file)
        load_button.grid(row=0, column=0, sticky=ctk.N+ctk.S)

        self.dark_mode = IntVar()
        self.dark_mode_switch = Checkbutton(frame, text="Dark Mode", variable=self.dark_mode, command=self.toggle_dark_mode)
        self.dark_mode_switch.grid(row=0, column=1, sticky=ctk.N+ctk.S)
        if(ctk.get_appearance_mode().lower() == 'dark'):
            self.dark_mode.set(1)
            self.toggle_dark_mode()

        self.root.mainloop()

    def temp(self, x, y):
        print(x, y)


    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        self.text_widget.configure(state='normal')
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.text_widget.delete('0.0', END)
                self.text_widget.insert(END, content)
        self.text_widget.configure(state="disable")

    def toggle_dark_mode(self):
        if self.dark_mode.get():
            self.root.configure(bg='black')
            # text_widget.configure(bg='black', fg='white', state='normal')
            ctk.set_appearance_mode('dark')
            # make dark_mode_switch also dark
            self.dark_mode_switch.configure(bg='black', fg='white', selectcolor='black')
        else:
            self.root.configure(bg='white')
            ctk.set_appearance_mode('light')
            self.dark_mode_switch.configure(bg='white', fg='black', selectcolor='white')

