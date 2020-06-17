from tkinter import *
import pickle

import tkinter as tk


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Text(self)
        self.button = tk.Button(self, text="Predict Genre", command=self.on_button)
        self.genre_classifier = pickle.load(open("classifier_lr.pkl", 'rb'))
        self.button.pack(side='bottom')
        self.entry.pack()
        self.labels = []

    def on_button(self):
        out = self.genre_classifier.predict([self.entry.get("1.0", 'end-1c')])[0]

        #self.button.destroy()
        for label in self.labels:
            label.destroy()
        label = Label(self, text= str(out))
        self.labels.append(label)
        label.pack()


app = SampleApp()
app.mainloop()
