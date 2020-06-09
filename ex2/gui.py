import tkinter as tk
from tkinter import filedialog
import skimage.io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

STARTING_COL_INDEX = 0
ANGLE_INDEX = 1
STARTING_FRAME_INDEX = 2



class ViewPoint:
    def __init__(self):
        self.root = tk.Tk()
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.title("Viewpoint GUI")
        self.text_box_vars = [tk.StringVar() for _ in range(3)]
        self.sliders = []
        self.images = []
        self.tk_images = []
        self.input_frame = None
        self.output_frame = None
        self.starting_frame_canvas = None
        self.controller_frame = None
        self.sliders_boxes_frame = None
        self.img_frame = None
        self.starting_frame = None
        self.starting_col = None
        self.ending_frame = None
        self.ending_col = None

    def choose_frames_loading(self):
        def save_and_close():
            self.starting_frame, self.starting_col = int(textbox_vars[0].get()), int(textbox_vars[1].get())
            self.ending_frame, self.ending_col = int(textbox_vars[2].get()),  int(textbox_vars[3].get())
            new_win.destroy()
            # TODO calculate the output to show

        textbox_vars = [tk.StringVar() for _ in range(4)]
        new_win = tk.Toplevel(self.root)
        tk.Label(new_win, text='Enter initial viewpoint:').grid(row=0, column=0, sticky='w', columnspan=4)
        # first frame
        tk.Label(new_win, text='Starting frame:').grid(row=1, column=0, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[0]).grid(row=1, column=1)
        # first col
        tk.Label(new_win, text='Starting column:').grid(row=1, column=2, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[1]).grid(row=1, column=3)

        # last Frame
        tk.Label(new_win, text='Ending frame:').grid(row=3, column=0, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[2]).grid(row=3, column=1)

        # last rol
        tk.Label(new_win, text='Ending column:').grid(row=3, column=2, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[3]).grid(row=3, column=3)
        tk.Button(new_win, text='Run', command=save_and_close).grid(row=4, column=0, columnspan=3)

        pass

    def update_text_box(self, val, index):
        self.text_box_vars[index].set(val)
        if index == STARTING_FRAME_INDEX and len(self.tk_images) != 0:
            self.starting_frame_canvas.create_image((150, 150), image=self.tk_images[int(val)])

    def update_slider(self, index):
        try:
            val = float(self.text_box_vars[index].get())
            self.sliders[index].set(val)
            if index == STARTING_FRAME_INDEX and len(self.tk_images) != 0:
                self.starting_frame_canvas.create_image((150, 150), image=self.tk_images[int(val)])
        except ValueError:
            pass

    def convert_to_uint8(self, image):
        img = image - np.min(image)
        return np.round(255 * img / np.max(img)).astype(np.uint8)

    def resize_image(self, image, max_width=300, max_height=300):
        # resize
        width, height = image.size
        if width > max_width:
            old_width = width
            width = max_width
            height = int(height * width / old_width)
        if height > max_height:
            old_height = height
            height = max_height
            width = int(width * height / old_height)
        image = image.resize((width, height), Image.ANTIALIAS)

        return ImageTk.PhotoImage(image)

    def load_images(self, path=None):
        if path is None:
            directory = filedialog.askdirectory()
        else:
            directory = path

        if directory != '':
            del self.tk_images[:]
            files = sorted(os.listdir(directory))

            first_img = skimage.io.imread(os.path.join(directory, files[0]))/255.0
            self.tk_images.append(self.resize_image(Image.fromarray(self.convert_to_uint8(first_img))))
            # update the sliders
            self.sliders[STARTING_FRAME_INDEX].configure(to=self.ending_frame)
            self.sliders[STARTING_COL_INDEX].configure(to=first_img.shape[1] - 1)
            images = np.zeros(np.insert(first_img.shape, 0, len(files)))
            images[0] = first_img
            # read all images
            for i in range(1,len(files)):
                images[i] = skimage.io.imread(os.path.join(directory, files[i]))/255.0
                self.tk_images.append(self.resize_image(Image.fromarray(self.convert_to_uint8(images[i]))))
            self.choose_frames_loading()

    def init_slider(self, frame, text, label_row, label_col, label_sticky, scale_from, scale_to,
                    resolution, index, color, scale_row, scale_col, scale_sticky):
        tk.Label(frame, text=text).grid(row=label_row, column=label_col, sticky=label_sticky)
        slider = tk.Scale(frame, orient='horizontal', from_=scale_from, to=scale_to,
                              resolution=resolution, command=lambda val: self.update_text_box(val, index),
                              cursor='DOT',
                              sliderlength=20, troughcolor=color, length=150)
        self.sliders.append(slider)
        slider.grid(row=scale_row, column=scale_col, sticky=scale_sticky)

    def init_box(self, frame, index, row, col):
        col_text_box = tk.Entry(frame, width=10, bd=3, textvariable=self.text_box_vars[
            index])
        col_text_box.grid(row=row, column=col, sticky='s')
        self.text_box_vars[STARTING_COL_INDEX].trace_add('write', lambda *args: self.update_slider(index))

    def run(self):

        # input frame
        tk.Label(self.root, text='Get input').grid(row=0, column=0, sticky='ew')
        self.input_frame = tk.Frame(self.root, relief='raised', borderwidth=3)
        tk.Checkbutton(self.input_frame, text='Make transfer motion', command=None).pack(side='right', padx=10, pady=10)
        tk.Button(self.input_frame, text='Load images',  command=self.load_images).pack(side='right',padx=60,
                                                                                 pady=25)
        # path input
        path_entry = tk.Entry(self.input_frame, bd=5, width=50)
        path_entry.pack(side='left')
        tk.Button(self.input_frame, text='Load', command=lambda: self.load_images(path_entry.get())).pack(
            side='left', padx=10, pady=20)
        self.input_frame.grid(row=1, column=0, sticky='ew')

        # controller frame
        tk.Label(self.root, text='Viewpoint controller').grid(row=2, column=0, sticky='ew')
        self.controller_frame = tk.Frame(self.root, relief='raised', borderwidth=3)
        self.controller_frame.grid(row=3, column=0, sticky='ew')

        # slider and box frame
        self.sliders_boxes_frame = tk.Frame(self.controller_frame, relief='raised', borderwidth=3)
        self.sliders_boxes_frame.grid(row=0, column=0, sticky='ew')
        tk.Label(self.controller_frame, text='Viewpoint by angle:').grid(row=0, column=0, sticky='nw')

        # starting frame image show
        self.img_frame = tk.Frame(self.controller_frame, borderwidth=3)
        self.img_frame.grid(row=0, column=1, sticky='nw')
        tk.Label(self.img_frame, text='Starting frame image:').grid(row=0, column=0, sticky='n')
        self.starting_frame_canvas = tk.Canvas(self.img_frame, height=300, width=300,borderwidth=3 ,
                                               relief='raised')
        self.starting_frame_canvas.grid(row=1, column=0, sticky='n')
        self.starting_frame_canvas.create_image((150, 150), image=None)

        # starting col slider
        self.init_slider(frame=self.sliders_boxes_frame, text='Starting column:', label_row=0, label_col=0,
                         label_sticky='we', scale_from=0, scale_to=10,
                         resolution=1, index=STARTING_COL_INDEX, color='pink', scale_row=1, scale_col=0,
                         scale_sticky='w')

        #  starting col text box
        self.init_box(frame=self.sliders_boxes_frame, index=STARTING_COL_INDEX, row=1, col=1)

        # angle slider
        self.init_slider(frame=self.sliders_boxes_frame, text='Angle:', label_row=2, label_col=0,
                         label_sticky='we', scale_from=1, scale_to=179,
                         resolution=0.1, index=ANGLE_INDEX, color='red', scale_row=3, scale_col=0,
                         scale_sticky='w')

        #  angle text box
        self.init_box(frame=self.sliders_boxes_frame, index=ANGLE_INDEX, row=3, col=1)

        # starting frame slider
        self.init_slider(frame=self.sliders_boxes_frame, text='Starting frame:', label_row=4, label_col=0,
                         label_sticky='we', scale_from=0, scale_to=20,
                         resolution=1, index=STARTING_FRAME_INDEX, color='blue', scale_row=5, scale_col=0,
                         scale_sticky='w')

        #  starting frame text box
        self.init_box(frame=self.sliders_boxes_frame, index=STARTING_FRAME_INDEX, row=5, col=1)

        # output
        tk.Label(self.root, text='Output result').grid(row=0, column=1, sticky='ewns')
        self.output_frame = tk.Frame(self.root, relief='raised', borderwidth=3, bg="blue")
        self.output_frame.grid(row=1, column=1, sticky='ewns', rowspan=8, columnspan=3)

        self.root.mainloop()


if __name__ == '__main__':
    view_point = ViewPoint()
    view_point.run()