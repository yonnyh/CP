import tkinter as tk
from tkinter import filedialog
import skimage.io
import os
import numpy as np
from PIL import Image, ImageTk
import ex2.light_field as lf
import ex2.area_marker as am

COL_INDEX = 0
ANGLE_INDEX = 1
FRAME_INDEX = 2

FOCUS_SLIDER = 0
FOCUS_RESOLUTION = 1


class ViewPoint:
    def __init__(self):
        self.root = tk.Tk()
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=100)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.title("Viewpoint GUI")
        self.text_box_vars = [tk.StringVar() for _ in range(3)]
        self.sliders = []
        self.tk_images = []
        self.input_frame = None
        self.output_frame = None
        self.controller_frame = None
        self.sliders_boxes_frame = None
        self.img_frame = None
        self.starting_frame_index = None
        self.starting_col_index = None
        self.ending_frame_index = None
        self.ending_col_index = None
        self.angle = None
        self.total_frame = None
        self.starting_frame_canvas = None
        self.output_label = None
        self.output_tk_image = None
        self.images = []
        self.translate_only = True
        self.lf_object = None
        self.uniform_height = 400
        self.total_cols = None

    def _not_translation_only(self):
        if self.translate_only:
            self.translate_only = False
        else:
            self.translate_only = True

    def _init_lf_object(self):
        self.lf_object = lf.LightFileViewPoint(self.images)
        if not self.translate_only:
            self.lf_object.apply_homographies_on_images()

    def choose_frames_loading(self):
        def save_and_close():
            # Get frames and columns from user
            self.starting_frame_index, self.starting_col_index = \
                int(textbox_vars[0].get()), int(textbox_vars[1].get())
            self.ending_frame_index, self.ending_col_index = \
                int(textbox_vars[2].get()), int(textbox_vars[3].get())

            # Initialize LightField object, and calculate its initial angle
            self._init_lf_object()
            self.angle = self.lf_object.frames_to_angle(
                self.starting_frame_index, self.starting_col_index,
                self.ending_frame_index, self.ending_col_index)

            # Update sliders and boxes
            self.update_text_box(self.starting_frame_index, FRAME_INDEX)
            self.update_text_box(self.starting_col_index, COL_INDEX)
            self.update_text_box(self.angle, ANGLE_INDEX)
            self.update_slider(FRAME_INDEX)
            self.update_slider(COL_INDEX)
            self.update_slider(ANGLE_INDEX)
            new_win.destroy()
            output = self.lf_object.calculate_view_point_by_angle(self.starting_frame_index, self.starting_col_index, self.angle)
            height, width = output.shape[0], output.shape[1]
            self.output_label.configure(height=height, width=width)
            self.output_tk_image = ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(output)),  self.uniform_height,  self.uniform_height * (width/height)))
            self.output_label.configure(image=self.output_tk_image)

        textbox_vars = [tk.StringVar() for _ in range(4)]
        new_win = tk.Toplevel(self.root)
        tk.Label(new_win, text='Enter initial viewpoint:').grid(row=0, column=0, sticky='w', columnspan=5)
        tk.Label(new_win, text='Total frames number: ').grid(row=1, column=0, sticky='w')
        tk.Label(new_win, width=6, text=str((self.total_frame - 1)), bg='red').grid(row=1, column=1)
        tk.Label(new_win, text='Total frame columns: ').grid(row=1, column=2, sticky='w')
        tk.Label(new_win, width=6, text=str((self.total_cols - 1)), bg='red').grid(row=1, column=3)

        # first frame
        tk.Label(new_win, text='Starting frame:').grid(row=2, column=0, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[0], bg='pink').grid(row=2, column=1)
        textbox_vars[0].set(0)
        # first col
        tk.Label(new_win, text='Starting column:').grid(row=2, column=2, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[1], bg='pink').grid(row=2, column=3)
        textbox_vars[1].set(0)

        # last Frame
        tk.Label(new_win, text='Ending frame:').grid(row=4, column=0, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[2], bg='cyan').grid(row=4, column=1)
        textbox_vars[2].set(self.total_frame - 1)

        # last col
        tk.Label(new_win, text='Ending column:').grid(row=4, column=2, sticky='w')
        tk.Entry(new_win, width=6, textvariable=textbox_vars[3], bg='cyan').grid(row=4, column=3)
        tk.Button(new_win, text='Run', command=save_and_close).grid(row=5, column=0, columnspan=3)
        textbox_vars[3].set(self.total_cols - 1)

    def update_text_box(self, val, index):
        self.text_box_vars[index].set(val)
        try:
            start_frame, start_col= int(self.text_box_vars[FRAME_INDEX].get()), int(self.text_box_vars[COL_INDEX].get())
            angle = float(self.text_box_vars[ANGLE_INDEX].get())
            # update output
            output = self.lf_object.calculate_view_point_by_angle(frame=start_frame, col=start_col, angle_deg=angle)
            height, width = output.shape[0], output.shape[1]
            self.output_label.configure(height=height, width=width)
            self.output_tk_image = ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(output)), int(self.uniform_height * (width / height)),
                self.uniform_height))
            self.output_label.configure(image=self.output_tk_image)
            if index == FRAME_INDEX and len(self.tk_images) != 0:
                self.starting_frame_canvas.create_image((150, 150), image=self.tk_images[int(val)])
            # if index == COL_INDEX and len(self.tk_images) != 0:
            #     col = int(val)
            #     self.starting_frame_canvas.create_line(col, 300, col, 0)
        except ValueError:
            pass
        except AttributeError:
            pass

    def update_slider(self, index):
        try:
            val = float(self.text_box_vars[index].get())
            self.sliders[index].set(val)
            start_frame, start_col = int(self.text_box_vars[FRAME_INDEX].get()), int(
                self.text_box_vars[COL_INDEX].get())
            angle = float(self.text_box_vars[ANGLE_INDEX].get())
            # update output
            output = self.lf_object.calculate_view_point_by_angle(frame=start_frame, col=start_col,
                                                                  angle_deg=angle)
            height, width = output.shape[0], output.shape[1]
            self.output_label.configure(height=height, width=width)
            self.output_tk_image = ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(output)), int(self.uniform_height * (width / height)),
                self.uniform_height))
            self.output_label.configure(image=self.output_tk_image)

            if index == FRAME_INDEX and len(self.tk_images) != 0:
                self.starting_frame_canvas.create_image((150, 150), image=self.tk_images[int(val)])
            # if index == COL_INDEX and len(self.tk_images) != 0:
            #     col = int(self.sliders[index].get())
            #     self.starting_frame_canvas.create_line(col, 300, col, 0, width=2)
        except ValueError:
            pass
        except AttributeError:
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
            height = height *(width / old_width)
        if height > max_height:
            old_height = height
            height = max_height
            width = width * (height / old_height)
        if int(width) <= 0:
            width = 1
        elif int(height) <= 0:
            height = 1
        image = image.resize((int(width), int(height)), Image.ANTIALIAS)

        return image

    def load_images(self, path=None):
        if path is None:
            directory = filedialog.askdirectory()
        else:
            directory = path

        if directory == '':
            raise IOError("Problem in loading images from this directory")

        del self.tk_images[:]
        files = sorted(os.listdir(directory))
        self.total_frame = len(files)
        first_img = skimage.io.imread(os.path.join(directory, files[0]))/255.0
        self.tk_images.append(ImageTk.PhotoImage(self.resize_image(Image.fromarray(self.convert_to_uint8(
            first_img)), 300, 300)))
        self.total_cols = first_img.shape[1]
        # update the sliders
        self.sliders[FRAME_INDEX].configure(to=self.total_frame - 1)
        self.sliders[COL_INDEX].configure(to=first_img.shape[1] - 1)
        self.images = np.zeros(np.insert(first_img.shape, 0, self.total_frame))
        self.images[0] = first_img

        # read all images
        for i in range(1, self.total_frame):
            self.images[i] = skimage.io.imread(os.path.join(directory, files[i]))/255.0
            self.tk_images.append(ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(self.images[i])))))

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
        text_box = tk.Entry(frame, width=10, bd=3, textvariable=self.text_box_vars[
            index])
        text_box.grid(row=row, column=col, sticky='s')
        self.text_box_vars[index].trace_add('write', lambda *args: self.update_slider(index))
        # todo: box has to create outputs as slider does

    def optimize_output(self):
        try:
            start_frame, start_col = int(self.text_box_vars[FRAME_INDEX].get()), int(
                self.text_box_vars[COL_INDEX].get())
            angle = float(self.text_box_vars[ANGLE_INDEX].get())
            # update output
            output = self.lf_object.calculate_view_point_by_angle(frame=start_frame, col=start_col,
                                                                  angle_deg=angle, fast=False)
            height, width = output.shape[0], output.shape[1]
            self.output_label.configure(height=height, width=width)
            self.output_tk_image = ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(output)), int(self.uniform_height * (width / height)),
                self.uniform_height))
            self.output_label.configure(image=self.output_tk_image)
        except AttributeError:
            pass

    def run(self):

        # input frame
        tk.Label(self.root, text='Get input').grid(row=0, column=0, sticky='we')
        self.input_frame = tk.Frame(self.root, relief='raised', borderwidth=3)
        tk.Checkbutton(self.input_frame, text='Make transfer motion',
                       command=self._not_translation_only).pack(side='bottom', padx=10, pady=10)
        tk.Button(self.input_frame, text='Load images',  command=self.load_images).pack(side='right',padx=5,
                                                                                 pady=5)
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
        self.init_slider(frame=self.sliders_boxes_frame, text='Starting column:', label_row=2, label_col=0,
                         label_sticky='we', scale_from=0, scale_to=10,
                         resolution=1, index=COL_INDEX, color='pink', scale_row=3, scale_col=0,
                         scale_sticky='w')

        #  starting col text box
        self.init_box(frame=self.sliders_boxes_frame, index=COL_INDEX, row=3, col=1)
        self.text_box_vars[COL_INDEX].set(0)


        # angle slider
        self.init_slider(frame=self.sliders_boxes_frame, text='Angle:', label_row=4, label_col=0,
                         label_sticky='we', scale_from=1, scale_to=179,
                         resolution=0.1, index=ANGLE_INDEX, color='red', scale_row=5, scale_col=0,
                         scale_sticky='w')

        #  angle text box
        self.init_box(frame=self.sliders_boxes_frame, index=ANGLE_INDEX, row=5, col=1)
        self.text_box_vars[ANGLE_INDEX].set(1.0)

        # starting frame slider
        self.init_slider(frame=self.sliders_boxes_frame, text='Starting frame:', label_row=0, label_col=0,
                         label_sticky='we', scale_from=0, scale_to=20,
                         resolution=1, index=FRAME_INDEX, color='blue', scale_row=1, scale_col=0,
                         scale_sticky='w')

        #  starting frame text box
        self.init_box(frame=self.sliders_boxes_frame, index=FRAME_INDEX, row=1, col=1)
        self.text_box_vars[FRAME_INDEX].set(0)

        # optimize output
        tk.Button(self.controller_frame, text='Optimize output', command=self.optimize_output).grid(row=1,
                                                                                                    column=0, sticky='ew')

        # output
        tk.Label(self.root, text='Output result').grid(row=0, column=1, sticky='ewns')
        self.output_label = tk.Label(self.root)
        self.output_label.grid(row=1, column=1, sticky='enws',
                               image=self.output_tk_image, rowspan=5)


        self.root.mainloop()


class Focus:
    def __init__(self):
        self.root = tk.Tk()
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.title("Focus GUI")
        self.text_box_vars = [tk.StringVar() for _ in range(2)]
        self.sliders = []
        self.images = []
        self.remove_occlusions = False
        self.input_frame = None
        self.output_frame = None
        self.img_frame = None
        self.ending_col = None
        self.output_label = None
        self.output_tk_image = None
        self.total_frame = None
        self.lf_object = None
        self.uniform_height = 700
        self.am_object = None
        self.left, self.right, self.up, self.down = None, None, None, None
    def _init_am_object(self):
        self.am_object = am.AreaMarker(self.images[(len(self.images) - 1)//2])

    def _init_lf_object(self):
        self.lf_object = lf.LightFieldRefocus(self.images)

    def update_remove_occlusions(self):
        if not self.remove_occlusions:
            self.remove_occlusions = True
            try:
                val = float(self.text_box_vars[FOCUS_SLIDER].get())
                self.show_result(val)
            except ValueError:
                pass
        else:
            self.remove_occlusions = False
            try:
                val = float(self.text_box_vars[FOCUS_SLIDER].get())
                self.show_result(val)
            except ValueError:
                pass

    def update_text_box(self, val, index):
        try:
            self.text_box_vars[index].set(val)
            self.show_result(val)
        except ValueError:
            pass

    def update_slider(self):
        try:
            val = float(self.text_box_vars[FOCUS_SLIDER].get())
            self.show_result(val)
            self.sliders[FOCUS_SLIDER].set(val)
        except ValueError:
            pass

    def update_resolution_slider(self):
        try:
            self.sliders[FOCUS_SLIDER].configure(resolution=float(self.text_box_vars[FOCUS_RESOLUTION].get()))
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
            height = height * width / old_width
        if height > max_height:
            old_height = height
            height = max_height
            width = width * height / old_height
        if int(width) <= 0:
            width = 1
        elif int(height) <= 0:
            height = 1
        return image.resize((int(width), int(height)), Image.ANTIALIAS)

    def show_result(self, shift_size=0.0, mark=False):
        try:
            if mark:
                output = self.lf_object.refocus_by_object(up=self.up, left=self.left, down=self.down,
                                                          right=self.right,
                                                          remove_occ=self.remove_occlusions)
            else:
                output = self.lf_object.refocus_by_shift(shift_size=shift_size,
                                                         remove_occ=self.remove_occlusions)
            height, width = output.shape[0], output.shape[1]
            self.output_label.configure(height=height, width=width)
            self.output_tk_image = ImageTk.PhotoImage(self.resize_image(Image.fromarray(
                self.convert_to_uint8(output)), self.uniform_height, self.uniform_height * (width / height)))
            self.output_label.configure(image=self.output_tk_image)
        except AttributeError:
            pass

    def load_images(self, path=None):
        if path is None:
            directory = filedialog.askdirectory()
        else:
            directory = path

        if directory == '':
            raise IOError("Problem in loading images from this directory")

        files = sorted(os.listdir(directory))
        self.total_frame = len(files)
        first_img = skimage.io.imread(os.path.join(directory, files[0])) / 255.0
        # update the sliders
        self.images = np.zeros(np.insert(first_img.shape, 0, self.total_frame))
        self.images[0] = first_img

        # read all images
        for i in range(1, self.total_frame):
            self.images[i] = skimage.io.imread(os.path.join(directory, files[i])) / 255.0

        # Initialize LightField object
        self._init_lf_object()
        self.show_result(float(self.text_box_vars[FOCUS_SLIDER].get()))

    def init_slider(self, frame, text, label_row, label_col, label_sticky, scale_from, scale_to,
                    resolution, index, color, scale_row, scale_col, scale_sticky):
        tk.Label(frame, text=text).grid(row=label_row, column=label_col, sticky=label_sticky)
        slider = tk.Scale(frame, orient='horizontal', from_=scale_from, to=scale_to,
                          resolution=resolution, command=lambda val: self.update_text_box(val, index),
                          cursor='DOT',
                          sliderlength=20, troughcolor=color, length=300)
        self.sliders.append(slider)
        slider.grid(row=scale_row, column=scale_col, sticky=scale_sticky)

    def init_box(self, frame, index, row, col,  sticky='s'):
        col_text_box = tk.Entry(frame, width=10, bd=3, textvariable=self.text_box_vars[
            index])
        col_text_box.grid(row=row, column=col, sticky=sticky)
        self.text_box_vars[index].trace_add('write', lambda *args: self.update_slider())

    def mark_object_and_show(self):
        try:
            self._init_am_object()
            self.left, self.right, self.up, self.down = self.am_object.get_square_from_user()
            self.show_result(mark=True)
            val = self.lf_object.best_focus
            self.update_text_box(val, FOCUS_SLIDER)
            self.update_slider()
        except AttributeError:
            pass

    def run(self):
        # input frame
        tk.Label(self.root, text='Get input').grid(row=0, column=0, sticky='ew')
        self.input_frame = tk.Frame(self.root, relief='raised', borderwidth=3)
        tk.Checkbutton(self.input_frame, text='Remove occlusions', command=self.update_remove_occlusions).pack(side='bottom',
                                                                                         padx=10, pady=10)
        tk.Button(self.input_frame, text='Load images', command=self.load_images).pack(side='right', padx=5,
                                                                                       pady=5)
        # path input
        path_entry = tk.Entry(self.input_frame, bd=5, width=50)
        path_entry.pack(side='left')
        tk.Button(self.input_frame, text='Load', command=lambda: self.load_images(path_entry.get())).pack(
            side='left', padx=10, pady=20)
        self.input_frame.grid(row=1, column=0, sticky='ew')

        # controller frame
        tk.Label(self.root, text='Focus controller').grid(row=2, column=0, sticky='ew')
        self.controller_frame = tk.Frame(self.root, relief='raised', borderwidth=3)
        self.controller_frame.grid(row=3, column=0)

        # mark object to focus
        tk.Button(self.controller_frame, text='Mark object to focus', command=self.mark_object_and_show,
                  borderwidth=5).grid(row=3, column=0, sticky='e')

        # focus slider
        self.init_slider(frame=self.controller_frame, text='Focus:', label_row=0, label_col=0,
                         label_sticky='w', scale_from=-5, scale_to=5,
                         resolution=0.1, index=FOCUS_SLIDER, color='pink', scale_row=1, scale_col=0,
                         scale_sticky='s')
        self.sliders[FOCUS_SLIDER].set(0.0)

        #  focus text box
        self.init_box(frame=self.controller_frame, index=FOCUS_SLIDER, row=1, col=1)
        self.text_box_vars[FOCUS_SLIDER].set(0.0)

        self.resolution_frame = tk.Frame(self.controller_frame)
        self.resolution_frame.grid(row=1, column=2)

        # resolution box
        tk.Label(self.resolution_frame, text='Slider resolution').grid(row=0, column=0)
        self.text_box_vars[FOCUS_RESOLUTION].set(0.1)

        resolution_text_box = tk.Entry(self.resolution_frame, width=10, bd=3, textvariable=self.text_box_vars[FOCUS_RESOLUTION])
        resolution_text_box.grid(row=1, column=0, sticky='s')
        self.text_box_vars[FOCUS_RESOLUTION].trace_add('write', lambda *args: self.update_resolution_slider())

        # output
        tk.Label(self.root, text='Output result').grid(row=0, column=1, sticky='ewns')
        self.output_label = tk.Label(self.root)
        self.output_label.grid(row=1, column=1, sticky='enws',
                               image=self.output_tk_image, rowspan=5)

        self.root.mainloop()


if __name__ == '__main__':
    # view_point = ViewPoint()
    # view_point.run()
    focus = Focus()
    focus.run()
